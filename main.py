from fastapi import FastAPI
import joblib
from pydantic import BaseModel, conlist
from typing import List
import numpy as np

app = FastAPI(title="Trade Analyser ML API", description="API for tade model", version="1.0")

# Declare Filenames
filename_model = "trade_analyser_model.joblib"
filename_scaler = "trade_analyser_scaler.joblib"


# load feature scaler and model
class Features(BaseModel):
    team1: List[conlist(float)]
    team2: List[conlist(float)]


loaded_model = joblib.load(filename_model)
loaded_scaler = joblib.load(filename_scaler)


@app.post("/predict", tags=["predictions"])
async def predict_emotion(features: Features):
    squad1 = np.array(features.team1)
    squad2 = np.array(features.team1)

    if squad1.ndim != 2:
        return {
            "error": "only 2 dimensional arrays allowed"
        }

    squad1_preds = loaded_model.predict(squad1).tolist()
    squad2_preds = loaded_model.predict(squad2).tolist()

    return {
        "first_squad": squad1_preds,
        "second_squad": squad2_preds
    }
