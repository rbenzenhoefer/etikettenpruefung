"""
Hybrid-Messung: Etikettenmaske -> Ober-/Unterkante -> Geometrie -> Schiefe alpha.

Masken-Quelle:
  --source gt     echte Masken aus Blender (data/masks/). Prüft NUR die Geometrie:
                  so gut kann das Verfahren höchstens werden, wenn das Netz perfekt wäre.
  --source model  Masken vom trainierten Netz (--model models/label_seg.pt).

Aufruf:
    python analysis/measure_from_mask.py --data data --source gt
    python analysis/measure_from_mask.py --data data --source model --model models/label_seg.pt --debug

Unterschied zur reinen Gradienten-Messung (measure_skew.py): Die Kante ist hier der
Übergang "nicht Etikett -> Etikett" in der Maske. Sicke, Deckelrand und Aufdruck
gehören nie zur Maske und können deshalb nicht mehr mit der Etikettenkante
verwechselt werden. Das Suchfenster darf dadurch viel größer sein (+-25 mm).
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

from measure_skew import (ALPHA_TOL_DEG, camera_basis, local_fit, pixel_to_cylinder,
                          px_per_mm, soll_curve, true_edge_height, wrap)

SEARCH_MM = 25.0   # großzügig: Schiefe summiert sich beim Rundumetikett auf bis zu ~+-10 mm


def mask_edge_points(prob, curve_cols, curve_rows, search_px, edge):
    """
    Je Bildspalte den Maskenübergang nahe der Soll-Kurve finden, subpixelgenau.
    edge="top":    Übergang außen -> Etikett (von oben nach unten gelesen)
    edge="bottom": Übergang Etikett -> außen
    Rückgabe: Liste (x, y) in kontinuierlichen Bildkoordinaten (Pixelmitte = i + 0.5).
    """
    h = prob.shape[0]
    pts = []
    for c in range(int(math.ceil(curve_cols[0])), int(curve_cols[-1])):
        x = c + 0.5
        r_exp = float(np.interp(x, curve_cols, curve_rows))
        y0 = int(max(r_exp - search_px, 0))
        y1 = int(min(r_exp + search_px, h - 1))
        col = prob[y0:y1 + 1, c]
        inside = col >= 0.5
        if edge == "top":
            idx = np.where(~inside[:-1] & inside[1:])[0]
        else:
            idx = np.where(inside[:-1] & ~inside[1:])[0]
        if not len(idx):
            continue
        # Pixelmitten liegen bei y0+i+0.5; nimm den Übergang, der der Soll-Kurve am nächsten ist
        i = idx[np.argmin(np.abs(y0 + idx + 1.0 - r_exp))]
        a, b = col[i], col[i + 1]
        t = (0.5 - a) / (b - a) if b != a else 0.5          # lineare Interpolation der 0,5-Schwelle
        pts.append((x, y0 + i + 0.5 + t))
    return pts


def measure_bucket(rec, data_dir, meta, get_prob, debug_dir=None):
    radius_m = meta["can_diameter_mm"] / 2000
    edges = {"top": meta["label_top_mm"], "bottom": meta["label_bottom_mm"]}
    fits = []
    for k, fname in enumerate(rec["images"]):
        prob = get_prob(rec, k)
        if prob is None:
            continue
        cam_meta = meta["cameras"][k]
        cam = camera_basis(cam_meta["position_mm"])
        az = math.radians(cam_meta["azimuth_deg"])
        vis = None
        if debug_dir is not None:
            img = cv2.imread(str(data_dir / "images" / fname))
            overlay = img.copy()
            overlay[prob >= 0.5] = (0.6 * overlay[prob >= 0.5] + 0.4 * np.array([255, 120, 0])).astype(np.uint8)
            vis = overlay

        for edge, z_soll in edges.items():
            ccols, crows = soll_curve(cam, az, z_soll, meta, radius_m)
            search = SEARCH_MM * px_per_mm(cam, az, z_soll, meta, radius_m)
            phi_l, z_l, pix = [], [], []
            for x, y in mask_edge_points(prob, ccols, crows, search, edge):
                P = pixel_to_cylinder(x, y, cam, meta, radius_m)
                if P is None:
                    continue
                phi_l.append(wrap(math.atan2(P[1], P[0]) - az))
                z_l.append(P[2] * 1000)
                pix.append((int(x), int(y)))
            fit = local_fit(phi_l, z_l, radius_m * 1000)
            if fit:
                fits.append({"cam": k, "edge": edge, "az_deg": cam_meta["azimuth_deg"],
                             **{key: v for key, v in fit.items() if key != "inlier_mask"}})
            if vis is not None:
                for c, r in zip(ccols.astype(int), crows.astype(int)):
                    cv2.circle(vis, (c, r), 1, (0, 200, 0), -1)                   # grün: Soll
                mask = fit["inlier_mask"] if fit else np.zeros(len(pix), bool)
                for (c, r), ok in zip(pix, mask):
                    if 0 <= r < vis.shape[0]:
                        vis[r, c] = (0, 0, 255) if ok else (0, 165, 255)          # rot/orange
        if vis is not None:
            cv2.imwrite(str(debug_dir / fname), vis)

    alphas = [f["alpha_deg"] for f in fits]
    return {"fits": fits, "alpha_deg_est": float(np.median(alphas)) if alphas else None}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--source", choices=["gt", "model"], default="gt")
    p.add_argument("--model", default="models/label_seg.pt")
    p.add_argument("--only-ids", type=str, default=None, help="z.B. 40-49 für die Val-Eimer")
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data)
    meta = json.loads((data_dir / "meta.json").read_text())
    records = [json.loads(l) for l in (data_dir / "labels.jsonl").read_text().splitlines() if l.strip()]
    if args.only_ids:
        lo, hi = map(int, args.only_ids.split("-"))
        records = [r for r in records if lo <= r["id"] <= hi]

    if args.source == "gt":
        def get_prob(rec, k):
            if not rec.get("masks"):
                return None
            m = cv2.imread(str(data_dir / "masks" / rec["masks"][k]), cv2.IMREAD_GRAYSCALE)
            return None if m is None else m.astype(np.float32) / 255.0
    else:
        from seg_model import get_device, load_model, predict_prob   # torch nur hier nötig
        device = get_device()
        model = load_model(args.model, device)

        def get_prob(rec, k):
            img = cv2.imread(str(data_dir / "images" / rec["images"][k]), cv2.IMREAD_COLOR)
            return None if img is None else predict_prob(model, img[:, :, ::-1].copy(), device)

    debug_dir = None
    if args.debug:
        debug_dir = data_dir / f"debug_{args.source}"
        debug_dir.mkdir(exist_ok=True)

    edges = {"top": meta["label_top_mm"], "bottom": meta["label_bottom_mm"]}
    err_alpha, err_height, decisions = [], [], []
    with open(data_dir / f"measurements_{args.source}.jsonl", "w") as out:
        for rec in records:
            res = measure_bucket(rec, data_dir, meta, get_prob, debug_dir)
            est = res["alpha_deg_est"]
            for f in res["fits"]:
                err_height.append(f["z_center_mm"] - true_edge_height(rec, meta, f["az_deg"], edges[f["edge"]]))
            if est is None:
                print(f"[{rec['id']:3d}] keine gültige Messung")
                continue
            err_alpha.append(est - rec["alpha_deg"])
            reject = abs(est) > ALPHA_TOL_DEG
            decisions.append((reject, abs(rec["alpha_deg"]) > ALPHA_TOL_DEG))
            print(f"[{rec['id']:3d}] alpha wahr {rec['alpha_deg']:+5.2f}°  gemessen {est:+5.2f}°  "
                  f"Fehler {est - rec['alpha_deg']:+5.2f}°  | {len(res['fits'])}/6 Fits  "
                  f"| {'RAUS' if reject else 'ok  '}")
            out.write(json.dumps({"id": rec["id"], "alpha_deg_est": est, "fits": res["fits"]}) + "\n")

    if not err_alpha:
        print("Keine Messungen.")
        return
    ea, eh = np.abs(err_alpha), np.abs(err_height)
    fp = sum(1 for rej, bad in decisions if rej and not bad)
    fn = sum(1 for rej, bad in decisions if not rej and bad)
    print(f"\nQuelle: {args.source}  |  {len(err_alpha)}/{len(records)} Eimer gemessen")
    print(f"Schiefe:    mittlerer Betragsfehler {ea.mean():.3f}°, Median {np.median(ea):.3f}°, max {ea.max():.3f}°")
    print(f"Kantenhöhe: mittlerer Betragsfehler {eh.mean():.2f} mm, max {eh.max():.2f} mm")
    print(f"Entscheidung (Toleranz {ALPHA_TOL_DEG}°): {fp} Fehlausschleusungen, {fn} Durchschlupf")


if __name__ == "__main__":
    main()