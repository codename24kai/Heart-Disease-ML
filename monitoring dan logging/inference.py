"""
Heart Disease Model Inference

Author : Kai
"""

from pathlib import Path

import joblib
import pandas as pd


class HeartDiseasePredictor:

    def __init__(self):

        base_dir = Path(__file__).resolve().parent

        self.model_path = (
            base_dir /
            "artifacts" /
            "models" /
            "best_model.pkl"
        )

        self.scaler_path = (
            base_dir /
            "preprocessing" /
            "scaler.pkl"
        )

        self.model = joblib.load(
            self.model_path
        )

        self.scaler = joblib.load(
            self.scaler_path
        )

        self.feature_names = [

            "age",
            "sex",
            "cp",
            "trestbps",
            "chol",
            "fbs",
            "restecg",
            "thalach",
            "exang",
            "oldpeak",
            "slope",
            "ca",
            "thal"

        ]

    # =====================================================
    # PREPROCESS
    # =====================================================

    def preprocess(
        self,
        input_data: dict
    ):

        dataframe = pd.DataFrame(
            [input_data]
        )

        dataframe = dataframe[
            self.feature_names
        ]

        dataframe = pd.DataFrame(

            self.scaler.transform(
                dataframe
            ),

            columns=self.feature_names

        )

        return dataframe

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(
        self,
        input_data: dict
    ):

        processed = self.preprocess(
            input_data
        )

        prediction = self.model.predict(
            processed
        )[0]

        probability = self.model.predict_proba(
            processed
        )[0]

        result = {

            "prediction": int(prediction),

            "label": (

                "Heart Disease"

                if prediction == 1

                else

                "No Heart Disease"

            ),

            "probability": {

                "negative": round(
                    float(probability[0]),
                    4
                ),

                "positive": round(
                    float(probability[1]),
                    4
                )

            }

        }

        return result


# =====================================================
# TEST
# =====================================================

if __name__ == "__main__":

    predictor = HeartDiseasePredictor()

    sample = {

        "age": 55,
        "sex": 1,
        "cp": 2,
        "trestbps": 130,
        "chol": 250,
        "fbs": 0,
        "restecg": 1,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 1.5,
        "slope": 2,
        "ca": 0,
        "thal": 2

    }

    result = predictor.predict(
        sample
    )

    print("\nPrediction Result\n")

    print(result)