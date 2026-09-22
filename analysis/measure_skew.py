"""
Misst Schiefe (alpha) und Höhenlage des Etiketts aus den Kamerabildern.

Logik pro Eimer, pro Kamera, jeweils für Ober- UND Unterkante:
  1. Soll-Kurve: Wo läge die Kante im Bild, wenn das Etikett perfekt sitzt?
     -> Soll-Kante (Kreis auf Höhe z_soll) per Kameramodell ins Bild projizieren.
  2. Kantensuche: je Bildspalte in einem Fenster um die Soll-Kurve alle Kanten
     (Gradienten-Peaks) suchen und die der Soll-Kurve NÄCHSTE nehmen.
     -> Aufdruck und Deckelrand werden so ignoriert, auch wenn sie kontrastreicher sind.
  3. Rückprojektion: Pixel -> Sichtstrahl -> Schnitt mit dem Zylinder -> (phi, z) in mm.
  4. Lokaler Fit je Kamera: z = a + b * phi   (phi relativ zur Kamerablickrichtung)
     Steigung b = r * tan(alpha) -> Schiefe;  a = Kantenhöhe in Blickrichtung.
     RANSAC ignoriert Ausreißer (Henkel, Naht, Reflexe).
  5. Kombinieren: Median über alle gültigen Einzelfits (3 Kameras x 2 Kanten).

Aufruf:
    python analysis/measure_skew.py --data data
    python analysis/measure_skew.py --data data --debug     # Overlay-Bilder nach data/debug/
"""

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import find_peaks
from sklearn.linear_model import RANSACRegressor

# ---------------------------------------------------------------- Parameter
MAX_VIEW_DEG = 50.0      # nur +-50° um die Kameraachse auswerten (Rand der Dose ist zu schräg)
WINDOW_MM = 7.0          # Suchfenster um die Soll-Kante (+-mm); Deckelrand liegt ~8 mm über der Oberkante
MIN_GRAD = 300.0         # Mindeststärke einer Kante (Sobel, 8-bit-Bild)
RANSAC_RESIDUAL_MM = 1.0 # Punkte weiter als 1 mm von der Geraden gelten als Ausreißer
MIN_POINTS = 30          # weniger Kantenpunkte -> Kamera/Kante verwerfen
MIN_INLIER_RATIO = 0.5
ALPHA_TOL_DEG = 0.5      # Entscheidung: |alpha| darüber -> ausschleusen


# ---------------------------------------------------------------- Kameramodell
def camera_basis(position_mm):
    """Kamera blickt waagerecht auf die Eimerachse (wie der Track-To-Constraint in Blender)."""
    C = np.array(position_mm, dtype=float) / 1000.0
    f = np.array([-C[0], -C[1], 0.0])
    f /= np.linalg.norm(f)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return C, r, u, f


def project(P, cam, meta):
    """Weltpunkt (m) -> Pixel (col, row)."""
    C, r, u, f = cam
    V = P - C
    depth = V @ f
    s = meta["camera_focal_mm"] / (meta["camera_sensor_width_mm"] / 2) * (meta["resolution_px"] / 2)
    half = meta["resolution_px"] / 2
    return (V @ r) / depth * s + half, -(V @ u) / depth * s + half


def pixel_to_cylinder(col, row, cam, meta, radius_m):
    """Pixel -> Sichtstrahl -> vorderer Schnittpunkt mit dem Zylinder (m) oder None."""
    C, r, u, f = cam
    s = meta["camera_focal_mm"] / (meta["camera_sensor_width_mm"] / 2) * (meta["resolution_px"] / 2)
    half = meta["resolution_px"] / 2
    D = (col - half) / s * r - (row - half) / s * u + f
    D /= np.linalg.norm(D)
    a = D[0] ** 2 + D[1] ** 2
    b = 2 * (C[0] * D[0] + C[1] * D[1])
    c = C[0] ** 2 + C[1] ** 2 - radius_m ** 2
    disc = b * b - 4 * a * c
    if disc < 0:
        return None
    t = (-b - math.sqrt(disc)) / (2 * a)   # kleineres t = der Kamera zugewandte Seite
    return C + t * D if t > 0 else None


def wrap(angle_rad):
    return (angle_rad + math.pi) % (2 * math.pi) - math.pi


# ---------------------------------------------------------------- Schritte 1-3
def soll_curve(cam, az_rad, z_mm, meta, radius_m):
    """Soll-Kante als Bildkurve: Spalten (aufsteigend) und zugehörige Zeilen."""
    phis = az_rad + np.radians(np.linspace(-MAX_VIEW_DEG, MAX_VIEW_DEG, 400))
    pts = [project(np.array([radius_m * math.cos(p), radius_m * math.sin(p), z_mm / 1000]), cam, meta)
           for p in phis]
    cols, rows = np.array(pts).T
    order = np.argsort(cols)
    return cols[order], rows[order]


def px_per_mm(cam, az_rad, z_mm, meta, radius_m):
    """Vertikaler Maßstab an der Kamerablickrichtung (Pixel pro mm Höhe)."""
    base = np.array([radius_m * math.cos(az_rad), radius_m * math.sin(az_rad), z_mm / 1000])
    _, r0 = project(base, cam, meta)
    _, r1 = project(base + [0, 0, 0.001], cam, meta)
    return abs(r1 - r0)


def find_edge_points(grad, curve_cols, curve_rows, window_px):
    """Je Spalte: der Soll-Kurve nächstgelegene Kante im Fenster."""
    h = grad.shape[0]
    out = []
    for col in range(int(math.ceil(curve_cols[0])), int(curve_cols[-1])):
        r_exp = np.interp(col, curve_cols, curve_rows)
        y0 = int(max(r_exp - window_px, 0))
        y1 = int(min(r_exp + window_px, h))
        peaks, _ = find_peaks(grad[y0:y1, col], height=MIN_GRAD)
        if len(peaks):
            best = peaks[np.argmin(np.abs(peaks + y0 - r_exp))]
            out.append((col, y0 + best))
    return out


# ---------------------------------------------------------------- Schritt 4
def local_fit(phi_local, z_mm, radius_mm):
    if len(phi_local) < MIN_POINTS:
        return None
    ransac = RANSACRegressor(residual_threshold=RANSAC_RESIDUAL_MM, random_state=0)
    ransac.fit(np.asarray(phi_local).reshape(-1, 1), np.asarray(z_mm))
    ratio = ransac.inlier_mask_.mean()
    if ratio < MIN_INLIER_RATIO:
        return None
    b = float(ransac.estimator_.coef_[0])
    return {"alpha_deg": math.degrees(math.atan(b / radius_mm)),
            "z_center_mm": float(ransac.estimator_.intercept_),
            "n": len(phi_local), "inlier_ratio": float(ratio),
            "inlier_mask": ransac.inlier_mask_}


# ---------------------------------------------------------------- pro Eimer
def measure_bucket(rec, data_dir, meta, debug=False):
    radius_m = meta["can_diameter_mm"] / 2000
    radius_mm = radius_m * 1000
    edges = {"top": meta["label_top_mm"], "bottom": meta["label_bottom_mm"]}
    fits = []

    for k, fname in enumerate(rec["images"]):
        gray = cv2.imread(str(data_dir / "images" / fname), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        cam_meta = meta["cameras"][k]
        cam = camera_basis(cam_meta["position_mm"])
        az = math.radians(cam_meta["azimuth_deg"])
        blur = cv2.GaussianBlur(gray, (3, 3), 0).astype(np.float32)
        grad = np.abs(cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=5))
        vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if debug else None

        for edge, z_soll in edges.items():
            ccols, crows = soll_curve(cam, az, z_soll, meta, radius_m)
            window = WINDOW_MM * px_per_mm(cam, az, z_soll, meta, radius_m)
            phi_l, z_l, pix = [], [], []
            for col, row in find_edge_points(grad, ccols, crows, window):
                P = pixel_to_cylinder(col, row, cam, meta, radius_m)
                if P is None:
                    continue
                phi_rel = wrap(math.atan2(P[1], P[0]) - az)     # relativ zur Kamera -> kein 180°-Sprung
                phi_l.append(phi_rel)
                z_l.append(P[2] * 1000)
                pix.append((col, row))

            fit = local_fit(phi_l, z_l, radius_mm)
            if fit:
                fits.append({"cam": k, "edge": edge, "az_deg": cam_meta["azimuth_deg"],
                             **{key: v for key, v in fit.items() if key != "inlier_mask"}})

            if debug:
                for c, r in zip(ccols.astype(int), crows.astype(int)):
                    cv2.circle(vis, (c, r), 1, (0, 200, 0), -1)            # grün: Soll
                mask = fit["inlier_mask"] if fit else np.zeros(len(pix), bool)
                for (c, r), ok in zip(pix, mask):
                    vis[r, c] = (0, 0, 255) if ok else (0, 165, 255)       # rot: Inlier, orange: Ausreißer
        if debug:
            (data_dir / "debug").mkdir(exist_ok=True)
            cv2.imwrite(str(data_dir / "debug" / fname), vis)

    alphas = [f["alpha_deg"] for f in fits]
    return {"fits": fits, "alpha_deg_est": float(np.median(alphas)) if alphas else None}


def true_edge_height(rec, meta, az_deg, z_soll):
    """Wahre Kantenhöhe in Kamerablickrichtung (aus den Blender-Labels)."""
    r = meta["can_diameter_mm"] / 2
    phi_label = wrap(math.radians(az_deg - rec["can_rot_deg"]))
    return z_soll + rec["offset_mm"] + r * math.tan(math.radians(rec["alpha_deg"])) * phi_label


# ---------------------------------------------------------------- Auswertung
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--debug", action="store_true")
    args = p.parse_args()
    data_dir = Path(args.data)
    meta = json.loads((data_dir / "meta.json").read_text())
    records = [json.loads(l) for l in (data_dir / "labels.jsonl").read_text().splitlines() if l.strip()]
    edges = {"top": meta["label_top_mm"], "bottom": meta["label_bottom_mm"]}

    err_alpha, err_height, decisions = [], [], []
    with open(data_dir / "measurements.jsonl", "w") as out:
        for rec in records:
            res = measure_bucket(rec, data_dir, meta, args.debug)
            est = res["alpha_deg_est"]
            for f in res["fits"]:
                err_height.append(f["z_center_mm"] - true_edge_height(rec, meta, f["az_deg"], edges[f["edge"]]))
            if est is None:
                print(f"[{rec['id']:3d}] keine gültige Messung")
                continue
            err_alpha.append(est - rec["alpha_deg"])
            reject = abs(est) > ALPHA_TOL_DEG
            decisions.append((reject, abs(rec["alpha_deg"]) > ALPHA_TOL_DEG))  # wahr schlecht = über derselben Toleranz
            print(f"[{rec['id']:3d}] alpha wahr {rec['alpha_deg']:+5.2f}°  gemessen {est:+5.2f}°  "
                  f"Fehler {est - rec['alpha_deg']:+5.2f}°  | {len(res['fits'])}/6 Fits  "
                  f"| {'RAUS' if reject else 'ok  '}")
            out.write(json.dumps({"id": rec["id"], "alpha_deg_est": est, "fits": res["fits"]}) + "\n")

    if not err_alpha:
        print("Keine Messungen.")
        return
    ea, eh = np.abs(err_alpha), np.abs(err_height)
    print(f"\n{len(err_alpha)}/{len(records)} Eimer gemessen")
    print(f"Schiefe:      mittlerer Betragsfehler {ea.mean():.3f}°, Median {np.median(ea):.3f}°, max {ea.max():.3f}°")
    print(f"Kantenhöhe:   mittlerer Betragsfehler {eh.mean():.2f} mm, max {eh.max():.2f} mm")
    fp = sum(1 for rej, bad in decisions if rej and not bad)
    fn = sum(1 for rej, bad in decisions if not rej and bad)
    print(f"Entscheidung (Toleranz {ALPHA_TOL_DEG}°): {fp} Fehlausschleusungen, {fn} Durchschlupf "
          f"bei {len(decisions)} Eimern")


if __name__ == "__main__":
    main()