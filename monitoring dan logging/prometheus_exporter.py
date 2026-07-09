"""
Prometheus Exporter

Author : Muhammad Keisa Nabhan
"""

from prometheus_client import (
    Counter,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST
)

from fastapi.responses import Response


# ======================================================
# PROMETHEUS METRICS
# ======================================================

REQUEST_COUNT = Counter(
    "prediction_requests_total",
    "Total prediction requests"
)

POSITIVE_PREDICTION = Counter(
    "prediction_positive_total",
    "Total positive predictions"
)

NEGATIVE_PREDICTION = Counter(
    "prediction_negative_total",
    "Total negative predictions"
)

PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds",
    "Prediction latency"
)


# ======================================================
# METRICS RESPONSE
# ======================================================

def metrics_response():

    return Response(
        content=generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )