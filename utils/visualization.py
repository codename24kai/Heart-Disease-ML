"""
Visualization Utilities

Author : Kai
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    precision_recall_curve
)


def save_visualizations(
    model,
    X_test,
    y_test,
    feature_names,
    output_dir
):
    """
    Save all visualization artifacts.
    """

    output_dir = Path(output_dir)

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:, 1]

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================

    cm = confusion_matrix(
        y_test,
        prediction
    )

    plt.figure(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False
    )

    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    importance = model.feature_importances_

    sorted_idx = importance.argsort()

    plt.figure(figsize=(8,6))

    plt.barh(
        range(len(sorted_idx)),
        importance[sorted_idx]
    )

    plt.yticks(
        range(len(sorted_idx)),
        feature_names[sorted_idx]
    )

    plt.title("Feature Importance")

    plt.tight_layout()

    plt.savefig(
        output_dir / "feature_importance.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # ROC CURVE
    # =====================================================

    from sklearn.metrics import roc_curve, auc

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {roc_auc:.3f}"
    )

    plt.plot(
        [0,1],
        [0,1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        output_dir / "roc_curve.png",
        dpi=300
    )

    plt.close()

    # =====================================================
    # PRECISION RECALL CURVE
    # =====================================================

    precision, recall, _ = precision_recall_curve(
        y_test,
        probability
    )

    plt.figure(figsize=(6,6))

    plt.plot(
        recall,
        precision
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.tight_layout()

    plt.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=300
    )

    plt.close()

    print("\nVisualization artifacts saved successfully.")