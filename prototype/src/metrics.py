"""Calcul des métriques médicales."""
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "accuracy": (tp + tn) / len(y_true),
        "sensibilite": tp / (tp + fn) if tp + fn else 0.0,   # kératocônes bien détectés
        "specificite": tn / (tn + fp) if tn + fp else 0.0,   # normaux bien reconnus
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def print_metrics(name, m):
    print(f"--- {name} ---")
    print(f"AUC          : {m['auc']:.3f}")
    print(f"Accuracy     : {m['accuracy']:.3f}")
    print(f"Sensibilité  : {m['sensibilite']:.3f}  (KC détectés)")
    print(f"Spécificité  : {m['specificite']:.3f}  (normaux reconnus)")
    print(f"Matrice      : TN={m['tn']} FP={m['fp']} | FN={m['fn']} TP={m['tp']}")
