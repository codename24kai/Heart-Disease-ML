from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "preprocessing"

TRAIN_PATH = DATA_DIR / "heart_train.csv"
TEST_PATH = DATA_DIR / "heart_test.csv"


def load_dataset():

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    return train_df, test_df


def prepare_data(train_df, test_df):

    X_train = train_df.drop(columns=["condition"])
    y_train = train_df["condition"]

    X_test = test_df.drop(columns=["condition"])
    y_test = test_df["condition"]

    return X_train, X_test, y_train, y_test