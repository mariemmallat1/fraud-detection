import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from src.data import SEED, load_clean_data, split_data
from src.features import build_preprocessor


def compute_metrics(y_true, proba, threshold=0.5):
    pred = (proba >= threshold).astype(int)
    return {
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "pr_auc": average_precision_score(y_true, proba),
    }


def build_baseline():
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])


if __name__ == "__main__":
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    model = build_baseline()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    proba = cross_val_predict(model, X_train, y_train, cv=cv,
                              method="predict_proba")[:, 1]

    for name, value in compute_metrics(y_train, proba).items():
        print(f"{name:10s} {value:.3f}")

    pred = (proba >= 0.5).astype(int)
    print("\nConfusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_train, pred))