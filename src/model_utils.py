import math
import numpy as np
from typing import Dict, Any


REQUIRED_FIELDS = {
    "amount",
    "timestamp"
}

ALLOWED_FIELDS = REQUIRED_FIELDS


def validate_schema(tx: Dict[str, Any]):

    missing = REQUIRED_FIELDS - tx.keys()
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    if not isinstance(tx["amount"], (int, float)):
        raise ValueError("Amount must be numeric")
    if not isinstance(tx["timestamp"], str):
        raise ValueError("Timestamp must be ISO string")

            


def extract_features(tx: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract raw features from a transaction.
    """
    amount = float(tx["amount"])
    timestamp = tx["timestamp"]

    # assume timestamp is ISO string
    hour = int(timestamp[11:13])

    return {
        "Amount": amount,
        "Hour": hour
    }


def apply_transformations(features: Dict[str, float]) -> Dict[str, float]:
    """apply feature tranformations"""
    hour = features['Hour']
    amount = features['Amount']

    hour_sin = math.sin(2 * math.pi * hour / 24)
    hour_cos = math.cos(2 * math.pi * hour / 24)

    return {
        "Amount": amount,
        "Hour_Sin": hour_sin,
        "Hour_Cos": hour_cos
    }


def assemble_feature_vector(transformed: Dict[str, float], feature_order: list[str]) -> np.ndarray:
    """
    Assemble feature vector in correct order.
    """
    vector = np.array(
        [transformed[name] for name in feature_order],
        dtype=np.float32
    ).reshape(1, -1)

    return vector


def preprocess(raw_transactions:Dict[str, Any], feature_order: list[str]):
    """
    Preprocess and transform the raw transaction data
    """
    validate_schema(raw_transactions)

    features = extract_features(raw_transactions)

    transformed = apply_transformations(features)
    
    vector = assemble_feature_vector(transformed, feature_order)

    return vector



def score_transaction(tx: Dict[str, Any], model, feature_order: list[str], threshold: float, model_version: str):
    vector = preprocess(tx, feature_order)

    fraud_prob = model.predict_proba(vector)[0, 1]
    is_fraud = fraud_prob >= threshold

    return {
        "fraud_probability": float(fraud_prob),
        "is_fraud": bool(is_fraud),
        "threshold": threshold,
        "model_version": model_version
    }
    