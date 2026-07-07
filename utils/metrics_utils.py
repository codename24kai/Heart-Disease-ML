"""
Evaluation Utilities

Author : Kai
"""

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    average_precision_score
)


def evaluate(model, X_test, y_test):
    """
    Evaluate classification model.

    Returns
    -------
    dict
        Dictionary containing evaluation results.
    """

    # =====================================================
    # Prediction
    # =====================================================

    prediction = model.predict(X_test)

    probability = model.predict_proba(X_test)[:, 1]

    # =====================================================
    # Metrics
    # =====================================================

    accuracy = accuracy_score(
        y_test,
        prediction
    )

    precision = precision_score(
        y_test,
        prediction
    )

    recall = recall_score(
        y_test,
        prediction
    )

    f1 = f1_score(
        y_test,
        prediction
    )

    report = classification_report(
        y_test,
        prediction
    )

    matrix = confusion_matrix(
        y_test,
        prediction
    )

    # =====================================================
    # ROC
    # =====================================================

    fpr, tpr, _ = roc_curve(
        y_test,
        probability
    )

    roc_auc = auc(
        fpr,
        tpr
    )

    # =====================================================
    # Precision Recall
    # =====================================================

    average_precision = average_precision_score(
        y_test,
        probability
    )

    # =====================================================
    # Return
    # =====================================================

    return {

        "accuracy": accuracy,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "roc_auc": roc_auc,

        "average_precision": average_precision,

        "report": report,

        "matrix": matrix,

        "prediction": prediction,

        "probability": probability,

        "fpr": fpr,

        "tpr": tpr

    }