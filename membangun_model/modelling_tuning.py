"""
Machine Learning Model Training with Hyperparameter Tuning
Heart Disease Cleveland Dataset

Author : Muhammad Kaisa Nabhan
"""

from pathlib import Path
import json
import warnings
import joblib

import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from utils.data_utils import (
    load_dataset,
    prepare_data
)

from utils.metrics_utils import (
    evaluate
)

from utils.visualization import (
    save_visualizations
)

warnings.filterwarnings("ignore")

# ==========================================================
# PATH
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

ARTIFACT_DIR = BASE_DIR / "artifacts"

REPORT_DIR = ARTIFACT_DIR / "reports"
MODEL_DIR = ARTIFACT_DIR / "models"
PARAM_DIR = ARTIFACT_DIR / "params"
METRIC_DIR = ARTIFACT_DIR / "metrics"

REPORT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

PARAM_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRIC_DIR.mkdir(
    parents=True,
    exist_ok=True
)

# ==========================================================
# BUILD GRID SEARCH MODEL
# ==========================================================

def build_grid_search():

    model = RandomForestClassifier(
        random_state=42
    )

    param_grid = {

        "n_estimators": [
            100,
            200,
            300
        ],

        "max_depth": [
            None,
            5,
            10,
            15
        ],

        "min_samples_split": [
            2,
            5,
            10
        ],

        "min_samples_leaf": [
            1,
            2,
            4
        ],

        "criterion": [
            "gini",
            "entropy"
        ]

    }

    grid_search = GridSearchCV(

        estimator=model,

        param_grid=param_grid,

        scoring="accuracy",

        cv=5,

        n_jobs=1,

        verbose=2,

        return_train_score=True

    )

    return grid_search

# ==========================================================
# MAIN
# ==========================================================

def main():

    print("=" * 60)
    print("Heart Disease Classification - Hyperparameter Tuning")
    print("=" * 60)

    mlflow.set_experiment(
        "Heart Disease Classification Tuning"
    )

    # ======================================================
    # LOAD DATA
    # ======================================================

    train_df, test_df = load_dataset()

    X_train, X_test, y_train, y_test = prepare_data(
        train_df,
        test_df
    )

    print(f"\nTraining Data : {X_train.shape}")
    print(f"Testing Data  : {X_test.shape}")

    # ======================================================
    # START MLFLOW RUN
    # ======================================================

    with mlflow.start_run():

        print("\nStarting Grid Search...\n")

        grid_search = build_grid_search()

        grid_search.fit(
            X_train,
            y_train
        )

        best_model = grid_search.best_estimator_

        print("\nGrid Search Finished")

        print("\nBest Parameters")

        for key, value in grid_search.best_params_.items():
            print(f"{key:<20}: {value}")

        print(f"\nBest CV Score : {grid_search.best_score_:.4f}")

        # ==================================================
        # EVALUATION
        # ==================================================

        result = evaluate(
            best_model,
            X_test,
            y_test
        )

        print("\n")
        print("=" * 60)
        print("MODEL EVALUATION")
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

        # ==================================================
        # SAVE BEST PARAMETER
        # ==================================================

        with open(
            PARAM_DIR / "best_params.json",
            "w"
        ) as file:

            json.dump(
                grid_search.best_params_,
                file,
                indent=4
            )

        # ==================================================
        # SAVE METRICS
        # ==================================================

        metrics = {

            "accuracy": float(result["accuracy"]),

            "precision": float(result["precision"]),

            "recall": float(result["recall"]),

            "f1_score": float(result["f1"]),

            "roc_auc": float(result["roc_auc"]),

            "average_precision": float(result["average_precision"]),

            "best_cv_score": float(grid_search.best_score_)

        }

        with open(
            METRIC_DIR / "evaluation_metrics.json",
            "w"
        ) as file:

            json.dump(
                metrics,
                file,
                indent=4
            )

        # ==================================================
        # SAVE REPORT
        # ==================================================

        with open(
            REPORT_DIR / "classification_report.txt",
            "w"
        ) as file:

            file.write(result["report"])

        pd.DataFrame(
            result["matrix"]
        ).to_csv(
            REPORT_DIR / "confusion_matrix.csv",
            index=False
        )

        # ==================================================
        # SAVE MODEL
        # ==================================================

        joblib.dump(
            best_model,
            MODEL_DIR / "best_model.pkl"
        )

        # ==================================================
        # SAVE VISUALIZATION
        # ==================================================
       
        save_visualizations(
            model=best_model,
            X_test=X_test,
            y_test=y_test,
            feature_names=X_train.columns.tolist(),
            output_dir=REPORT_DIR
        )
        
                # ==================================================
        # MLFLOW - LOG PARAMETERS
        # ==================================================

        mlflow.log_params(grid_search.best_params_)

        # ==================================================
        # MLFLOW - LOG METRICS
        # ==================================================

        mlflow.log_metric(
            "accuracy",
            result["accuracy"]
        )

        mlflow.log_metric(
            "precision",
            result["precision"]
        )

        mlflow.log_metric(
            "recall",
            result["recall"]
        )

        mlflow.log_metric(
            "f1_score",
            result["f1"]
        )

        mlflow.log_metric(
            "roc_auc",
            result["roc_auc"]
        )

        mlflow.log_metric(
            "average_precision",
            result["average_precision"]
        )

        mlflow.log_metric(
            "best_cv_score",
            grid_search.best_score_
        )

        # ==================================================
        # MLFLOW - LOG MODEL
        # ==================================================

        mlflow.sklearn.log_model(
            sk_model=best_model,
            artifact_path="model"
        )

        # ==================================================
        # MLFLOW - LOG ARTIFACTS
        # ==================================================

        mlflow.log_artifacts(
            str(ARTIFACT_DIR)
        )

        # ==================================================
        # SUMMARY
        # ==================================================

        print("\n")
        print("=" * 60)
        print("ARTIFACT SUMMARY")
        print("=" * 60)

        print(f"Best CV Score       : {grid_search.best_score_:.4f}")
        print(f"Accuracy            : {result['accuracy']:.4f}")
        print(f"Precision           : {result['precision']:.4f}")
        print(f"Recall              : {result['recall']:.4f}")
        print(f"F1 Score            : {result['f1']:.4f}")
        print(f"ROC AUC             : {result['roc_auc']:.4f}")
        print(f"Average Precision   : {result['average_precision']:.4f}")

        print("\nGenerated Artifacts\n")

        for file in ARTIFACT_DIR.rglob("*"):

            if file.is_file():

                print(file.relative_to(BASE_DIR))

        print("\n")
        print("=" * 60)
        print("MLFLOW")
        print("=" * 60)

        print("Experiment :", "Heart Disease Classification Tuning")
        print("Status     : Success")

        print("\n")
        print("=" * 60)
        print("TRAINING FINISHED SUCCESSFULLY")
        print("=" * 60)


if __name__ == "__main__":
    main()