"""
Machine Learning Model Training
Heart Disease Cleveland Dataset

Author : Muhammad Kaisa Nabhan
"""

from pathlib import Path
import warnings

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier

from utils.data_utils import (
    load_dataset,
    prepare_data
)

from utils.metrics_utils import (
    evaluate
)

warnings.filterwarnings("ignore")

# ==========================================================
# PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

# ==========================================================
# BUILD MODEL
# ==========================================================

def build_model():

    model = RandomForestClassifier(

        n_estimators=100,

        random_state=42

    )

    return model


# ==========================================================
# TRAIN MODEL
# ==========================================================

def train_model(

    model,

    X_train,

    y_train

):

    model.fit(

        X_train,

        y_train

    )

    return model


# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("Heart Disease Classification")
    print("=" * 60)

    mlflow.set_experiment(

        "Heart Disease Classification"

    )

    mlflow.sklearn.autolog()

    train_df, test_df = load_dataset()

    X_train, X_test, y_train, y_test = prepare_data(

        train_df,

        test_df

    )

    print(f"Training Data : {X_train.shape}")
    print(f"Testing Data  : {X_test.shape}")

    with mlflow.start_run():

        model = build_model()

        model = train_model(

            model,

            X_train,

            y_train

        )

        result = evaluate(

            model,

            X_test,

            y_test

        )

        print("\n")
        print("=" * 60)
        print("Evaluation Result")
        print("=" * 60)

        print(f"Accuracy            : {result['accuracy']:.4f}")
        print(f"Precision           : {result['precision']:.4f}")
        print(f"Recall              : {result['recall']:.4f}")
        print(f"F1 Score            : {result['f1']:.4f}")
        print(f"ROC AUC             : {result['roc_auc']:.4f}")
        print(f"Average Precision   : {result['average_precision']:.4f}")

        print("\nClassification Report\n")

        print(result["report"])

        print("\nConfusion Matrix\n")

        print(result["matrix"])
        
        # ==========================================================
        # PRINT TRAINING INFORMATION
        # ==========================================================

        print("\n")
        print("=" * 60)
        print("Training Information")
        print("=" * 60)

        print("Model               : RandomForestClassifier")
        print("Number of Trees     : 100")
        print("Random State        : 42")

        print("\n")
        print("=" * 60)
        print("MLflow")
        print("=" * 60)

        print("Experiment Tracking : Enabled")
        print("Autolog             : Enabled")

        print("\n")
        print("=" * 60)
        print("Training Finished Successfully")
        print("=" * 60)


if __name__ == "__main__":
    main()