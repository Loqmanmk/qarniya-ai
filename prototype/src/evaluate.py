"""Évalue le modèle sauvegardé sur le jeu de TEST (jamais vu pendant l'entraînement).
Lancer :  python src/evaluate.py"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import roc_curve
from torch.utils.data import DataLoader

import config as C
from dataset import CornOrbDataset, get_splits
from metrics import compute_metrics, print_metrics
from model import QarniyaNet
from train import predict


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_df = get_splits()
    model = QarniyaNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(C.MODEL_PATH, map_location=device))

    dl = DataLoader(CornOrbDataset(test_df), batch_size=C.BATCH_SIZE, num_workers=2)
    ys, ps = predict(model, dl, device)
    m = compute_metrics(ys, ps)
    print(f"Test : {len(test_df)} yeux")
    print_metrics("TEST", m)

    # sauvegarde des résultats + courbe ROC (utile pour les slides)
    (C.OUT_DIR / "test_metrics.json").write_text(json.dumps(m, indent=2))
    test_df.assign(prob_kc=ps).to_csv(C.OUT_DIR / "test_predictions.csv", index=False)
    fpr, tpr, _ = roc_curve(ys, ps)
    plt.figure(figsize=(4, 4))
    plt.plot(fpr, tpr, label=f"AUC = {m['auc']:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("1 - Spécificité"); plt.ylabel("Sensibilité"); plt.title("Courbe ROC (test)")
    plt.legend(); plt.tight_layout(); plt.savefig(C.OUT_DIR / "roc_test.png", dpi=150)
    print(f"\nFichiers écrits dans {C.OUT_DIR}: test_metrics.json, test_predictions.csv, roc_test.png")


if __name__ == "__main__":
    main()
