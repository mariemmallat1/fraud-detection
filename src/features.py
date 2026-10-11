from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler

COLS_TO_SCALE = ["Amount", "Time"]


def build_preprocessor():
    """Scale Amount and Time; leave the V1-V28 columns unchanged."""
    return ColumnTransformer(
        transformers=[("scale", RobustScaler(), COLS_TO_SCALE)],
        remainder="passthrough",
    )


if __name__ == "__main__":
    from src.data import load_clean_data, split_data

    df = load_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    prep = build_preprocessor()
    prep.fit(X_train)  # learns median/spread from TRAIN only

    print("Amount median learned on train:",
          prep.named_transformers_["scale"].center_[0])

    X_train_p = prep.transform(X_train)
    X_test_p = prep.transform(X_test)  # applies the train numbers to test
    print("Train shape:", X_train_p.shape)
    print("Test shape: ", X_test_p.shape)