"""Prédiction pour UN œil + carte de chaleur Grad-CAM sur chacune des 4 cartes.

Exemple :
  python src/predict.py --folder data/cornorb/ORBSCAN_Dataset/9C6SM/OD
  -> écrit outputs/9C6SM_OD_gradcam.png
"""
import argparse
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageOps

import config as C
from dataset import eval_transform, load_eye
from model import QarniyaNet

_model = None


def get_model(device):
    global _model
    if _model is None:
        _model = QarniyaNet(pretrained=False).to(device)
        _model.load_state_dict(torch.load(C.MODEL_PATH, map_location=device))
        _model.eval()
    return _model


def risk_level(p):
    if p < 0.3:
        return "Faible"
    if p < 0.7:
        return "À surveiller"
    return "Élevé"


def analyse_eye(img_paths, eye="OD"):
    """img_paths : 4 chemins dans l'ordre C.MAPS. Retourne (probabilité, [4 overlays RGB])."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(device)
    imgs = load_eye(img_paths, eye)
    tf = eval_transform()
    x = torch.stack([tf(im) for im in imgs]).unsqueeze(0).to(device)  # (1,4,3,H,W)

    # --- Grad-CAM : gradients de la sortie par rapport à la dernière couche conv ---
    feats = {}
    h = model.features.register_forward_hook(lambda m, i, o: feats.setdefault("a", o))
    logit = model(x)
    h.remove()
    act = feats["a"]                                   # (4, 512, h, w)
    grad = torch.autograd.grad(logit.sum(), act)[0]    # (4, 512, h, w)
    weights = grad.mean(dim=(2, 3), keepdim=True)
    cams = torch.relu((weights * act).sum(1)).detach().cpu().numpy()  # (4, h, w)
    cams /= cams.max() + 1e-8                          # même échelle pour les 4 cartes

    prob = torch.sigmoid(logit).item()
    overlays = []
    for im, cam in zip(imgs, cams):
        if eye == "OS":                                # remettre l'image dans son sens d'origine
            im, cam = ImageOps.mirror(im), cam[:, ::-1]
        # fond en niveaux de gris pour que la chaleur ne se mélange pas aux couleurs de l'appareil
        base = np.array(im.convert("L").convert("RGB")).astype(np.float32)
        heat = cv2.resize(np.ascontiguousarray(cam), (base.shape[1], base.shape[0]))
        color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)[:, :, ::-1]
        alpha = (0.65 * heat)[..., None]               # transparent là où le modèle ne regarde pas
        overlays.append((base * (1 - alpha) + color * alpha).astype(np.uint8))
    return prob, overlays


def paths_from_folder(folder):
    folder = Path(folder)
    out = []
    for m in C.MAPS:
        found = list(folder.glob(f"*{m}*.png")) + list(folder.glob(f"*{m}*.jpg"))
        if not found:
            raise FileNotFoundError(f"Carte '{m}' introuvable dans {folder}")
        out.append(found[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="dossier contenant les 4 cartes d'un œil")
    ap.add_argument("--eye", default=None, help="OD ou OS (déduit du nom du dossier sinon)")
    args = ap.parse_args()

    folder = Path(args.folder)
    eye = args.eye or ("OS" if folder.name.upper() == "OS" else "OD")
    prob, overlays = analyse_eye(paths_from_folder(folder), eye)

    print(f"Score de risque kératocône : {prob:.1%}  ->  {risk_level(prob)}")
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.4))
    for ax, ov, name in zip(axes, overlays, C.MAPS):
        ax.imshow(ov); ax.set_title(name); ax.axis("off")
    fig.suptitle(f"{folder.parent.name} {eye} — risque {prob:.1%} ({risk_level(prob)})")
    C.OUT_DIR.mkdir(exist_ok=True)
    out = C.OUT_DIR / f"{folder.parent.name}_{eye}_gradcam.png"
    fig.tight_layout(); fig.savefig(out, dpi=110)
    print("Image écrite :", out)


if __name__ == "__main__":
    main()
