import time

import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.data import SEED, load_clean_data, split_data
from src.features import build_preprocessor
from src.train import compute_metrics

# Illustrative assumptions, NOT real bank figures
COST_FN = 100
COST_FP = 2


def get_models(y_train):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()  # about 600
    return {
        "logreg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=100, class_weight="balanced_subsample",
            n_jobs=-1, random_state=SEED),
        "xgboost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            scale_pos_weight=ratio, eval_metric="aucpr",
            tree_method="hist", n_jobs=-1, random_state=SEED),
        "lightgbm": LGBMClassifier(
            n_estimators=300, learning_rate=0.05,
            scale_pos_weight=ratio, n_jobs=-1,
            random_state=SEED, verbose=-1),
    }


if __name__ == "__main__":
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    rows = []
    for name, clf in get_models(y_train).items():
        start = time.time()
        pipe = Pipeline([("prep", build_preprocessor()), ("model", clf)])
        proba = cross_val_predict(pipe, X_train, y_train, cv=cv,
                                  method="predict_proba")[:, 1]
        m = compute_metrics(y_train, proba)
        tn, fp, fn, tp = confusion_matrix(y_train, proba >= 0.5).ravel()
        m.update({"FP": fp, "FN": fn,
                  "cost@0.5": COST_FN * fn + COST_FP * fp,
                  "minutes": round((time.time() - start) / 60, 1)})
        rows.append({"model": name, **m})
        print(f"{name} done", flush=True)

    table = pd.DataFrame(rows).set_index("model").round(3)
    table = table.sort_values("pr_auc", ascending=False)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", None)
    print("\n", table.to_string())

    from pathlib import Path
    Path("reports").mkdir(exist_ok=True)
    table.to_csv("reports/model_comparison.csv")
    print("\nSaved to reports/model_comparison.csv")