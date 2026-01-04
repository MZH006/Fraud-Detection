from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
from model_utils import score_transaction

artifact = joblib.load("../artifacts/fraud_model_v1_lgr.pkl")
MODEL = artifact["model"]
FEATURE_ORDER = artifact["features"]
THRESHOLD = artifact["threshold"]
MODEL_VERSION = artifact["model_type"]

app = FastAPI(title="Credit Card Fraud Detection API")


class Transaction(BaseModel):
    amount: float
    timestamp: str


@app.post("/predict")
def predict(tx: Transaction):
    try:
        tx_dict = tx.model_dump()

        result = score_transaction(
            tx=tx_dict,
            model=MODEL,
            feature_order=FEATURE_ORDER,
            threshold=THRESHOLD,
            model_version=MODEL_VERSION
        )

        return result
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

