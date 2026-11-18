"""Paramètres communs du projet Qarniya AI."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # racine du dépôt
DATA_DIR = ROOT / "data" / "cornorb"
IMG_DIR = DATA_DIR / "ORBSCAN_Dataset"
CSV_PATH = DATA_DIR / "clinical_data_and_labels.csv"
OUT_DIR = ROOT / "outputs"
MODEL_PATH = OUT_DIR / "qarniya_model.pt"

# Les 4 cartes d'un œil, toujours dans cet ordre
MAPS = ["Axial", "Anterior", "Posterior", "Pachymetry"]

IMG_SIZE = 256        # les PNG font 540x540, on réduit pour aller vite
BATCH_SIZE = 16
EPOCHS = 25
LR = 3e-4
VAL_FOLD = 1          # le fold 1 du train sert de validation
SEED = 42
