"""
Trainiert das U-Net auf den Blender-Bildern + Etikettenmasken.

Aufruf:
    python analysis/train_seg.py --data data --epochs 30

Wichtig:
  - Aufteilung Train/Val nach EIMER (nicht nach Bild), sonst sieht das Netz im
    Val-Set denselben Eimer aus anderer Kamera und die Werte sind geschönt.
  - Training auf Ausschnitten (Crops) in voller Auflösung, bevorzugt um die
    Etikettenkante herum: Dort entscheidet sich die Messgenauigkeit, und
    Herunterskalieren würde genau diese Kantenschärfe kosten.
  - Validierung auf dem ganzen Bild, Metrik: IoU der Maske.
"""

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from seg_model import build_model, get_device, to_tensor


class LabelSegDataset(Dataset):
    def __init__(self, data_dir, records, crop, train):
        self.items = [(data_dir / "images" / im, data_dir / "masks" / m)
                      for r in records for im, m in zip(r["images"], r["masks"])]
        self.crop, self.train = crop, train

    def __len__(self):
        return len(self.items)

    def _edge_biased_crop(self, img, mask):
        h, w = mask.shape
        c = self.crop
        edges = np.argwhere(cv2.Canny((mask > 0.5).astype(np.uint8) * 255, 50, 150) > 0)
        if len(edges) and random.random() < 0.7:          # 70 %: Ausschnitt um eine Kante
            cy, cx = edges[random.randrange(len(edges))]
            cy += random.randint(-c // 3, c // 3)
            cx += random.randint(-c // 3, c // 3)
        else:                                              # 30 %: beliebiger Ausschnitt
            cy, cx = random.randrange(h), random.randrange(w)
        y0 = int(np.clip(cy - c // 2, 0, h - c))
        x0 = int(np.clip(cx - c // 2, 0, w - c))
        return img[y0:y0 + c, x0:x0 + c], mask[y0:y0 + c, x0:x0 + c]

    def _augment(self, img):
        img = img.astype(np.float32)
        img = img * random.uniform(0.7, 1.3) + random.uniform(-20, 20)      # Helligkeit/Kontrast
        img = img * np.array([random.uniform(0.93, 1.07) for _ in range(3)])  # leichter Farbstich
        if random.random() < 0.3:
            img = cv2.GaussianBlur(img, (0, 0), random.uniform(0.3, 1.2))    # Unschärfe
        img = img + np.random.normal(0, random.uniform(0, 6), img.shape)     # Sensorrauschen
        return np.clip(img, 0, 255).astype(np.uint8)

    def __getitem__(self, i):
        img_path, mask_path = self.items[i]
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)[:, :, ::-1]      # BGR -> RGB
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
        if self.train:
            img, mask = self._edge_biased_crop(img, mask)
            if random.random() < 0.5:                                      # Spiegeln ist hier geometrisch harmlos
                img, mask = img[:, ::-1], mask[:, ::-1]
            img = self._augment(img)
        return to_tensor(np.ascontiguousarray(img)), torch.from_numpy(np.ascontiguousarray(mask))[None]


def dice_loss(logits, target, eps=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2, 3))
    return (1 - (2 * inter + eps) / (p.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps)).mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ious = []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        pred = torch.sigmoid(model(x)) > 0.5
        gt = y > 0.5
        inter = (pred & gt).sum().item()
        union = (pred | gt).sum().item()
        ious.append(inter / union if union else 1.0)
    return float(np.mean(ious))


def pad32(x):
    h, w = x.shape[-2:]
    return F.pad(x, (0, (32 - w % 32) % 32, 0, (32 - h % 32) % 32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--crop", type=int, default=384)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--encoder", default="resnet18")
    p.add_argument("--out", default="models/label_seg.pt")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    data_dir = Path(args.data)
    records = [json.loads(l) for l in (data_dir / "labels.jsonl").read_text().splitlines() if l.strip()]
    records = [r for r in records if r.get("masks")]
    if not records:
        raise SystemExit("Keine Masken gefunden - Datensatz mit dem neuen generate_dataset.py erzeugen.")

    random.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    val_recs, train_recs = records[:n_val], records[n_val:]
    print(f"{len(train_recs)} Eimer Training, {len(val_recs)} Eimer Validierung "
          f"(je {len(records[0]['images'])} Kamerabilder)")

    device = get_device()
    print(f"Gerät: {device}")
    train_dl = DataLoader(LabelSegDataset(data_dir, train_recs, args.crop, True),
                          batch_size=args.batch, shuffle=True, num_workers=2, drop_last=True,
                          persistent_workers=True)
    val_dl = DataLoader(LabelSegDataset(data_dir, val_recs, args.crop, False), batch_size=1)

    model = build_model(args.encoder, pretrained=True).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0, losses = time.time(), []
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = F.binary_cross_entropy_with_logits(logits, y) + dice_loss(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()

        model.eval()
        val_iou = evaluate(model, ((pad32(x), pad32(y)) for x, y in val_dl), device)
        flag = ""
        if val_iou > best:
            best = val_iou
            torch.save({"encoder": args.encoder, "state_dict": model.state_dict(),
                        "val_iou": val_iou, "epoch": epoch}, out_path)
            flag = "  <- gespeichert"
        print(f"Epoche {epoch:3d}/{args.epochs}  Loss {np.mean(losses):.4f}  "
              f"Val-IoU {val_iou:.4f}  ({time.time() - t0:.0f} s){flag}")

    print(f"Bestes Modell (Val-IoU {best:.4f}) in {out_path}")


if __name__ == "__main__":
    main()