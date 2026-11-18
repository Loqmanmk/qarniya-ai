"""Entraînement du modèle.  Lancer :  python src/train.py   (option : --epochs 3 pour un test rapide)"""
import argparse
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import config as C
from dataset import CornOrbDataset, get_splits
from metrics import compute_metrics, print_metrics
from model import QarniyaNet


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    probs, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(x.to(device)))
        probs += p.float().cpu().tolist(); ys += y.tolist()
    return np.array(ys), np.array(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=C.EPOCHS)
    ap.add_argument("--limit", type=int, default=0, help="nb d'yeux max (test rapide)")
    args = ap.parse_args()

    set_seed(C.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Appareil :", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "")

    train_df, val_df, _ = get_splits()
    if args.limit:
        train_df = train_df.sample(min(args.limit, len(train_df)), random_state=C.SEED)
        val_df = val_df.sample(min(args.limit // 2, len(val_df)), random_state=C.SEED)
    print(f"Train : {len(train_df)} yeux  |  Validation : {len(val_df)} yeux")

    train_dl = DataLoader(CornOrbDataset(train_df, train=True), batch_size=C.BATCH_SIZE,
                          shuffle=True, num_workers=2, pin_memory=True, persistent_workers=True)
    val_dl = DataLoader(CornOrbDataset(val_df), batch_size=C.BATCH_SIZE, num_workers=2)

    model = QarniyaNet().to(device)
    # poids de classe : il y a moins de kératocônes que de normaux
    pos_weight = torch.tensor((train_df.label == 0).sum() / (train_df.label == 1).sum(), device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    C.OUT_DIR.mkdir(exist_ok=True)
    best_auc = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in tqdm(train_dl, desc=f"Epoch {epoch}/{args.epochs}", leave=False):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total += loss.item() * len(y)
        sched.step()

        ys, ps = predict(model, val_dl, device)
        m = compute_metrics(ys, ps)
        print(f"Epoch {epoch:2d}  loss={total / len(train_df):.4f}  val_AUC={m['auc']:.3f}  "
              f"sens={m['sensibilite']:.3f}  spec={m['specificite']:.3f}")
        if m["auc"] > best_auc:
            best_auc = m["auc"]
            torch.save(model.state_dict(), C.MODEL_PATH)
            print(f"   -> meilleur modèle sauvegardé ({C.MODEL_PATH.name})")

    print(f"\nTerminé. Meilleure AUC validation : {best_auc:.3f}")
    print("Étape suivante : python src/evaluate.py")


if __name__ == "__main__":
    main()
