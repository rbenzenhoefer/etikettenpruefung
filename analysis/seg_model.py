"""
Segmentierungsmodell für die Etikettenmaske (von Training und Messung gemeinsam genutzt).

U-Net mit ImageNet-vortrainiertem ResNet18-Encoder (segmentation_models_pytorch).
Ausgabe: 1 Kanal, Logit "Pixel gehört zum Etikett".
"""

import numpy as np
import torch
import segmentation_models_pytorch as smp

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")        # Apple-Silicon-GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(encoder="resnet18", pretrained=True):
    return smp.Unet(encoder_name=encoder,
                    encoder_weights="imagenet" if pretrained else None,
                    in_channels=3, classes=1)


def to_tensor(img_rgb):
    """uint8 RGB (H,W,3) -> normalisierter Tensor (3,H,W)."""
    x = (img_rgb.astype(np.float32) - MEAN) / STD
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def load_model(path, device):
    ckpt = torch.load(path, map_location=device)
    model = build_model(ckpt["encoder"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval()


@torch.no_grad()
def predict_prob(model, img_rgb, device):
    """Wahrscheinlichkeitskarte (H,W) in [0,1]. Das Bild wird auf ein Vielfaches von 32 gepolstert."""
    h, w = img_rgb.shape[:2]
    ph, pw = (32 - h % 32) % 32, (32 - w % 32) % 32
    padded = np.pad(img_rgb, ((0, ph), (0, pw), (0, 0)), mode="edge")
    x = to_tensor(padded)[None].to(device)
    prob = torch.sigmoid(model(x))[0, 0, :h, :w]
    return prob.float().cpu().numpy()