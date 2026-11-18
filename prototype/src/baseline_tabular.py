"""Point de comparaison simple : régression logistique sur les chiffres du CSV
(Kmax, épaisseur minimale, astigmatisme, asphéricité). Lancer : python src/baseline_tabular.py"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config as C
from metrics import compute_metrics, print_metrics

FEATURES = ["kmax_value_D", "pachy_thinnest_um", "pachy_central_um", "astig_value_D",
            "asphericity_anterior", "asphericity_posterior"]

from dataset import load_table

df = load_table().dropna(subset=FEATURES)  # mêmes yeux que le modèle images
train, test = df[df.split_80_20 == "train"], df[df.split_80_20 == "test"]
clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
clf.fit(train[FEATURES], train.label)
print_metrics("Baseline tabulaire (TEST)", compute_metrics(test.label, clf.predict_proba(test[FEATURES])[:, 1]))
