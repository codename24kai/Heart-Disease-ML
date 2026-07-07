"""
FastAPI Model Serving

Author : Kai
"""

from fastapi import FastAPI
from pydantic import BaseModel

from inference import HeartDiseasePredictor

from time import time

from prometheus_client import Counter
from prometheus_client import Histogram
from prometheus_client import generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

from fastapi.responses import Response


# ======================================================
# LOAD MODEL
# ======================================================

predictor = HeartDiseasePredictor()

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

    REQUEST_COUNT.inc()

    start = time()

    result = predictor.predict(

        data.model_dump()

    )

    elapsed = time() - start

    PREDICTION_LATENCY.observe(

        elapsed

    )

    if result["prediction"] == 1:

        POSITIVE_PREDICTION.inc()

    else:

        NEGATIVE_PREDICTION.inc()

    return result

@app.get("/metrics")

def metrics():

    return Response(

        generate_latest(),

        media_type=CONTENT_TYPE_LATEST

    )