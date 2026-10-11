from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

RAW_PATH = Path("data/raw/creditcard.csv")
SEED = 42


def load_clean_data(path=RAW_PATH):
    """Load the CSV and remove duplicate rows."""
    df = pd.read_csv(path)
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def split_data(df, test_size=0.2, seed=SEED):
    """Split into train/test, keeping the same fraud share in both."""
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )


if __name__ == "__main__":
    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)
    print("Train:", X_train.shape, "| frauds:", y_train.sum(),
          f"({y_train.mean():.4%})")
    print("Test: ", X_test.shape, "| frauds:", y_test.sum(),
          f"({y_test.mean():.4%})")