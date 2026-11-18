"""Chargement des données CornOrb : 4 cartes PNG par œil + étiquette (0 normal, 1 kératocône)."""
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms as T

from config import CSV_PATH, IMG_DIR, IMG_SIZE, MAPS, VAL_FOLD

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # normalisation ImageNet


def map_path(patient, eye, m, img_dir=IMG_DIR):
    return img_dir / patient / eye / f"{patient}_{eye}_{m}.png"


def load_table():
    """Lit le CSV et garde seulement les yeux qui ont bien leurs 4 images."""
    df = pd.read_csv(CSV_PATH, dtype={"patient_code": str})
    has_imgs = df.apply(
        lambda r: all(map_path(r.patient_code, r.eye, m).exists() for m in MAPS), axis=1
    )
    missing = (~has_imgs).sum()
    if missing:
        print(f"[info] {missing} yeux sans images ignorés")
    # 46 yeux apparaissent 2 fois dans le CSV avec des valeurs différentes mais un seul
    # jeu d'images : impossible de savoir quelle ligne correspond -> on les exclut.
    dup = df.duplicated(["patient_code", "eye"], keep=False)
    if dup.any():
        print(f"[info] {dup.sum()} lignes en double (même patient + œil) ignorées")
    return df[has_imgs & ~dup].reset_index(drop=True)


def get_splits():
    """Utilise le découpage officiel du dataset (par patient, pas de fuite entre ensembles)."""
    df = load_table()
    test = df[df.split_80_20 == "test"]
    train_all = df[df.split_80_20 == "train"]
    val = train_all[train_all.fold == VAL_FOLD]
    train = train_all[train_all.fold != VAL_FOLD]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def load_eye(img_paths, eye):
    """Ouvre les 4 cartes. Les yeux gauches (OS) sont retournés en miroir
    pour que nasal/temporal soient du même côté que pour l'œil droit (OD)."""
    imgs = []
    for p in img_paths:
        im = Image.open(p).convert("RGB")
        if eye == "OS":
            im = ImageOps.mirror(im)
        imgs.append(im)
    return imgs


def eval_transform():
    return T.Compose([T.Resize((IMG_SIZE, IMG_SIZE)), T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform():
    # Pas de changement de couleur : la couleur EST l'information (échelle de l'appareil)
    return T.Compose([
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomAffine(degrees=8, translate=(0.04, 0.04), scale=(0.95, 1.05)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])


class CornOrbDataset(Dataset):
    def __init__(self, df, train=False):
        self.df = df
        self.tf = train_transform() if train else eval_transform()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        paths = [map_path(r.patient_code, r.eye, m) for m in MAPS]
        imgs = load_eye(paths, r.eye)
        x = torch.stack([self.tf(im) for im in imgs])  # (4, 3, H, W)
        y = torch.tensor(float(r.label))
        return x, y
