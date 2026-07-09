"""
FastAPI Model Serving

Author : Kai
"""

from time import time

from fastapi import FastAPI
from pydantic import BaseModel

from inference import HeartDiseasePredictor

from prometheus_exporter import (
    REQUEST_COUNT,
    POSITIVE_PREDICTION,
    NEGATIVE_PREDICTION,
    PREDICTION_LATENCY,
    metrics_response
)

# ======================================================
# LOAD MODEL
# ======================================================

predictor = HeartDiseasePredictor()

# ======================================================
# FASTAPI
# ======================================================

app = FastAPI(

    title="Heart Disease Prediction API",

    description="Machine Learning API for Heart Disease Classification",

    version="1.0.0"

)

# ======================================================
# INPUT SCHEMA
# ======================================================

class HeartInput(BaseModel):

    age: float
    sex: int
    cp: int
    trestbps: float
    chol: float
    fbs: int
    restecg: int
    thalach: float
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


# ======================================================
# ROOT
# ======================================================

@app.get("/")
def root():

    return {

        "message": "Heart Disease Prediction API",

        "status": "Running"

    }


# ======================================================
# HEALTH CHECK
# ======================================================

@app.get("/health")
def health():

    return {

        "status": "healthy"

    }


# ======================================================
# PREDICT
# ======================================================

@app.post("/predict")
def predict(data: HeartInput):

    start = time()

    try:

        result = predictor.predict(

            data.model_dump()

        )

    finally:

        elapsed = time() - start

        REQUEST_COUNT.inc()

        PREDICTION_LATENCY.observe(

            elapsed

        )

    if result["prediction"] == 1:

        POSITIVE_PREDICTION.inc()

    else:

        NEGATIVE_PREDICTION.inc()

    return result


# ======================================================
# PROMETHEUS METRICS
# ======================================================

@app.get("/metrics")
def metrics():

    return metrics_response()