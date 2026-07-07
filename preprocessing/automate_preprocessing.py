"""
Automated Data Preprocessing
Heart Disease Cleveland Dataset

Author : Muhammad Keisa Nabhan
"""

import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ======================================================
# PATH
# ======================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "heart_cleveland_upload.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "preprocessing"
)

TRAIN_PATH = os.path.join(
    OUTPUT_DIR,
    "heart_train.csv"
)

TEST_PATH = os.path.join(
    OUTPUT_DIR,
    "heart_test.csv"
)

SCALER_PATH = os.path.join(
    OUTPUT_DIR,
    "scaler.pkl"
)


# ======================================================
# LOAD DATA
# ======================================================

def load_data(path):
    """
    Load dataset
    """

    df = pd.read_csv(path)

    print("=" * 50)
    print("Dataset Loaded Successfully")
    print(f"Shape : {df.shape}")
    print("=" * 50)

    return df


# ======================================================
# CHECK MISSING VALUE
# ======================================================

def check_missing(df):

    print("\nMissing Value")

    print(df.isnull().sum())

    return df


# ======================================================
# REMOVE DUPLICATE
# ======================================================

def remove_duplicate(df):

    duplicate = df.duplicated().sum()

    print(f"\nDuplicate Data : {duplicate}")

    df = df.drop_duplicates()

    print(f"Shape After Remove Duplicate : {df.shape}")

    return df


# ======================================================
# PREPROCESS
# ======================================================

def preprocess(df):

    X = df.drop(columns=["condition"])

    y = df["condition"]

    return X, y


# ======================================================
# SPLIT DATA
# ======================================================

def split_data(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


# ======================================================
# SCALING
# ======================================================

def scaling(X_train, X_test):

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, SCALER_PATH)

    print("\nScaler Saved Successfully")
    print(SCALER_PATH)

    X_train_scaled = pd.DataFrame(
        X_train_scaled,
        columns=X_train.columns
    )

    X_test_scaled = pd.DataFrame(
        X_test_scaled,
        columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled


# ======================================================
# SAVE DATA
# ======================================================

def save_dataset(
    X_train,
    X_test,
    y_train,
    y_test
):

    train = X_train.copy()
    train["condition"] = y_train.reset_index(drop=True)

    test = X_test.copy()
    test["condition"] = y_test.reset_index(drop=True)

    train.to_csv(
        TRAIN_PATH,
        index=False
    )

    test.to_csv(
        TEST_PATH,
        index=False
    )

    print("\nDataset Saved Successfully")
    print(TRAIN_PATH)
    print(TEST_PATH)

# ======================================================
# MAIN
# ======================================================

def main():

    df = load_data(DATASET_PATH)

    df = check_missing(df)

    df = remove_duplicate(df)

    X, y = preprocess(df)

    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test = scaling(
        X_train,
        X_test
    )

    save_dataset(
        X_train,
        X_test,
        y_train,
        y_test
    )


if __name__ == "__main__":
    main()