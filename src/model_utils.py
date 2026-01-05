import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
import logging

from feature_store.store import FeatureStore

logger = logging.getLogger(__name__)

REQUIRED_FIELDS = {
    "customer_id",
    "terminal_id", 
    "amount",
    "timestamp"
}



def validate_schema(tx: Dict[str, Any]):
    """Validate incoming transaction has required fields"""
    missing = REQUIRED_FIELDS - tx.keys()
    if missing:
        raise ValueError(f"Missing required fields: {missing}")
    
    if not isinstance(tx["amount"], (int, float)):
        raise ValueError("Amount must be numeric")
    if not isinstance(tx["timestamp"], str):
        raise ValueError("Timestamp must be ISO string")
    if not isinstance(tx["customer_id"], (int, str)):
        raise ValueError("Customer ID must be int or string")
    if not isinstance(tx["terminal_id"], (int, str)):
        raise ValueError("Terminal ID must be int or string")

            


def extract_realtime_features(tx: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract features available in real-time.
    """
    amount = float(tx["amount"])
    timestamp = tx["timestamp"]
    
    # Parse timestamp
    dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    
    hour = dt.hour
    day_of_week = dt.weekday()
    
    # Temporal features
    features = {
        'TX_HOUR_SIN': math.sin(2 * math.pi * hour / 24),
        'TX_HOUR_COS': math.cos(2 * math.pi * hour / 24),
        'TX_DAY_OF_WEEK': day_of_week,
        'TX_IS_WEEKEND': 1 if day_of_week >= 5 else 0,
        'TX_IS_NIGHT': 1 if (hour >= 22 or hour <= 6) else 0,
        
        # Amount features
        'TX_AMOUNT': amount,
        'TX_AMOUNT_HIGH': 1 if amount > 200 else 0,
        'TX_NIGHT_HIGH_AMOUNT': 1 if ((hour >= 22 or hour <= 6) and amount > 200) else 0,
    }
    
    return features

def enrich_with_feature_store(
    tx: Dict[str, Any], 
    realtime_features: Dict[str, float],
    feature_store: FeatureStore
) -> Dict[str, float]:
    """
    Enrich real-time features with customer and terminal features from feature store.
    """
    customer_id = int(tx["customer_id"])
    terminal_id = int(tx["terminal_id"])
    amount = float(tx["amount"])
    
    # Fetch customer features
    customer_features = feature_store.get_customer_features(customer_id)
    
    # Fetch terminal features
    terminal_features = feature_store.get_terminal_features(terminal_id)
    
    # Combine all features
    all_features = {**realtime_features, **customer_features, **terminal_features}
    
    # Compute derived features
    # Customer amount deviation
    avg_amount_7d = customer_features['CUSTOMER_AVG_AMOUNT_7D']
    std_amount_7d = customer_features['CUSTOMER_STD_AMOUNT_7D']
    
    if std_amount_7d > 0:
        amount_deviation = (amount - avg_amount_7d) / std_amount_7d
    else:
        amount_deviation = 0.0
    
    all_features['CUSTOMER_AMOUNT_DEVIATION'] = np.clip(amount_deviation, -10, 10)
    all_features['CUSTOMER_HIGH_DEVIATION'] = 1 if abs(amount_deviation) > 2 else 0
    
    # Risk score
    risk_score = (
        terminal_features['TERMINAL_HIGH_RISK'] * 2 +
        all_features['CUSTOMER_HIGH_DEVIATION'] * 3 +
        all_features['TX_NIGHT_HIGH_AMOUNT']
    )
    all_features['RISK_SCORE'] = risk_score
    
    return all_features




def assemble_feature_vector(features: Dict[str, float], feature_order: list[str]) -> np.ndarray:
    """Assemble feature vector in correct order for model"""
    vector = np.array(
        [features.get(name, 0.0) for name in feature_order],
        dtype=np.float32
    ).reshape(1, -1)
    
    return vector

def score_transaction(
    tx: Dict[str, Any], 
    model, 
    feature_order: list[str], 
    threshold: float, 
    model_version: str,
    feature_store: Optional[FeatureStore] = None
) -> Dict[str, Any]:
    """
    Score a transaction for fraud using the feature store.
    
    Args:
        tx: Transaction dict with customer_id, terminal_id, amount, timestamp
        model: Trained model
        feature_order: List of feature names in correct order
        threshold: Decision threshold
        model_version: Model identifier
        feature_store: FeatureStore instance (optional, will create if not provided)
        
    Returns:
        Dict with fraud_probability, is_fraud, risk_level, etc.
    """
    # Validate
    validate_schema(tx)
    
    # Initialize feature store if not provided
    if feature_store is None:
        feature_store = FeatureStore()
    
    # Extract real-time features
    realtime_features = extract_realtime_features(tx)
    
    #  Enrich with feature store
    all_features = enrich_with_feature_store(tx, realtime_features, feature_store)
    
    # Assemble feature vector
    vector = assemble_feature_vector(all_features, feature_order)
    
    # Predict
    fraud_prob = float(model.predict_proba(vector)[0, 1])
    is_fraud = fraud_prob >= threshold
    
    # Risk level categorization
    if fraud_prob >= 0.9:
        risk_level = "CRITICAL"
    elif fraud_prob >= 0.7:
        risk_level = "HIGH"
    elif fraud_prob >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"
    
    return {
        "transaction_id": tx.get("transaction_id", f"tx_{datetime.now().timestamp()}"),
        "customer_id": tx["customer_id"],
        "terminal_id": tx["terminal_id"],
        "amount": tx["amount"],
        "fraud_probability": fraud_prob,
        "is_fraud": is_fraud,
        "risk_level": risk_level,
        "threshold": threshold,
        "model_version": model_version,
        "timestamp": tx["timestamp"],
        "features_used": {
            "customer_avg_amount_7d": all_features.get('CUSTOMER_AVG_AMOUNT_7D'),
            "terminal_fraud_rate_7d": all_features.get('TERMINAL_FRAUD_RATE_7D'),
            "amount_deviation": all_features.get('CUSTOMER_AMOUNT_DEVIATION'),
            "risk_score": all_features.get('RISK_SCORE')
        }
    }
    