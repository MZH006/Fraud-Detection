"""
Model utilities for fraud detection including feature extraction and scoring.
Provides functions for real-time feature engineering and transaction scoring.
"""

import math
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from datetime import datetime, timedelta
from numpy.typing import NDArray

from config.logging_config import get_logger
from src.feature_store.store import FeatureStore

logger = get_logger(__name__)

REQUIRED_FIELDS = {
    "customer_id",
    "terminal_id", 
    "amount",
    "timestamp"
}



def validate_schema(tx: Dict[str, Any]) -> None:
    """
    Validate incoming transaction has required fields and correct types.

    Args:
        tx: Transaction dictionary to validate

    Raises:
        ValueError: If required fields are missing or types are incorrect

    Example:
        >>> tx = {"customer_id": 123, "terminal_id": 456, "amount": 100.0, "timestamp": "2024-01-15T14:30:00"}
        >>> validate_schema(tx)  # No error if valid
    """
    missing = REQUIRED_FIELDS - tx.keys()
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if not isinstance(tx["amount"], (int, float)):
        raise ValueError("Amount must be numeric")
    if tx["amount"] < 0:
        raise ValueError("Amount must be non-negative")
    if not isinstance(tx["timestamp"], str):
        raise ValueError("Timestamp must be ISO string")
    if not isinstance(tx["customer_id"], (int, str)):
        raise ValueError("Customer ID must be int or string")
    if not isinstance(tx["terminal_id"], (int, str)):
        raise ValueError("Terminal ID must be int or string")

            


def extract_realtime_features(tx: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract features available in real-time from a transaction.

    Computes temporal features (hour, day of week with cyclical encoding),
    amount-based features, and interaction features that don't require
    historical data.

    Args:
        tx: Transaction dictionary containing:
            - amount (float): Transaction amount
            - timestamp (str): ISO format timestamp (e.g., "2024-01-15T14:30:00")

    Returns:
        Dictionary of extracted features including:
            - TX_HOUR_SIN, TX_HOUR_COS: Cyclical hour encoding
            - TX_DAY_OF_WEEK: Day of week (0=Monday, 6=Sunday)
            - TX_IS_WEEKEND: Binary flag for weekend transactions
            - TX_IS_NIGHT: Binary flag for night transactions (10pm-6am)
            - TX_AMOUNT: Transaction amount
            - TX_AMOUNT_HIGH: Binary flag for high amounts (>$200)
            - TX_NIGHT_HIGH_AMOUNT: Interaction feature for risky combinations

    Raises:
        ValueError: If timestamp is invalid or cannot be parsed

    Example:
        >>> tx = {"amount": 150.0, "timestamp": "2024-01-15T14:30:00"}
        >>> features = extract_realtime_features(tx)
        >>> "TX_AMOUNT" in features
        True
        >>> features["TX_IS_NIGHT"]
        0
    """
    amount = float(tx["amount"])
    timestamp = tx["timestamp"]

    # Parse timestamp - handle both with and without timezone
    try:
        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format: {timestamp}") from e

    hour = dt.hour
    day_of_week = dt.weekday()

    # Temporal features with cyclical encoding to preserve hour continuity
    features = {
        'TX_HOUR_SIN': math.sin(2 * math.pi * hour / 24),
        'TX_HOUR_COS': math.cos(2 * math.pi * hour / 24),
        'TX_DAY_OF_WEEK': float(day_of_week),
        'TX_IS_WEEKEND': 1.0 if day_of_week >= 5 else 0.0,
        'TX_IS_NIGHT': 1.0 if (hour >= 22 or hour <= 6) else 0.0,

        # Amount features
        'TX_AMOUNT': amount,
        'TX_AMOUNT_HIGH': 1.0 if amount > 200 else 0.0,
        'TX_NIGHT_HIGH_AMOUNT': 1.0 if ((hour >= 22 or hour <= 6) and amount > 200) else 0.0,
    }

    return features

def enrich_with_feature_store(
    tx: Dict[str, Any],
    realtime_features: Dict[str, float],
    feature_store: FeatureStore
) -> Dict[str, float]:
    """
    Enrich real-time features with customer and terminal historical features.

    Fetches customer and terminal aggregations from the feature store and
    computes derived features like amount deviation and risk scores.

    Args:
        tx: Transaction dictionary containing customer_id, terminal_id, amount
        realtime_features: Features extracted from current transaction
        feature_store: FeatureStore instance for fetching historical features

    Returns:
        Dictionary containing all features (realtime + historical + derived):
            - All realtime features
            - Customer aggregations (transaction counts, avg amounts, std dev)
            - Terminal aggregations (fraud rates, transaction counts)
            - Derived features (amount deviation, risk score)

    Example:
        >>> tx = {"customer_id": 100, "terminal_id": 500, "amount": 300.0, ...}
        >>> realtime_features = extract_realtime_features(tx)
        >>> all_features = enrich_with_feature_store(tx, realtime_features, feature_store)
        >>> "CUSTOMER_AVG_AMOUNT_7D" in all_features
        True
    """
    customer_id = int(tx["customer_id"])
    terminal_id = int(tx["terminal_id"])
    amount = float(tx["amount"])

    # Fetch customer features from feature store
    customer_features = feature_store.get_customer_features(customer_id)

    # Fetch terminal features from feature store
    terminal_features = feature_store.get_terminal_features(terminal_id)

    # Combine all features
    all_features = {**realtime_features, **customer_features, **terminal_features}

    # Compute derived features

    # 1. Customer amount deviation (z-score)
    avg_amount_7d = customer_features['CUSTOMER_AVG_AMOUNT_7D']
    std_amount_7d = customer_features['CUSTOMER_STD_AMOUNT_7D']

    if std_amount_7d > 0:
        amount_deviation = (amount - avg_amount_7d) / std_amount_7d
    else:
        amount_deviation = 0.0

    # Clip to prevent extreme outliers from dominating
    all_features['CUSTOMER_AMOUNT_DEVIATION'] = float(np.clip(amount_deviation, -10, 10))
    all_features['CUSTOMER_HIGH_DEVIATION'] = 1.0 if abs(amount_deviation) > 2 else 0.0

    # 2. Combined risk score (weighted sum of risk indicators)
    risk_score = (
        terminal_features['TERMINAL_HIGH_RISK'] * 2 +  # Terminal has high fraud rate
        all_features['CUSTOMER_HIGH_DEVIATION'] * 3 +   # Amount is unusual for customer
        all_features['TX_NIGHT_HIGH_AMOUNT']            # High amount at night
    )
    all_features['RISK_SCORE'] = float(risk_score)

    return all_features




def assemble_feature_vector(features: Dict[str, float], feature_order: list[str]) -> NDArray[np.float32]:
    """
    Assemble feature vector in the correct order expected by the model.

    Args:
        features: Dictionary of feature name to value
        feature_order: List of feature names in the order expected by model

    Returns:
        2D numpy array of shape (1, n_features) ready for model prediction

    Example:
        >>> features = {"feature_1": 1.0, "feature_2": 2.0, "feature_3": 3.0}
        >>> feature_order = ["feature_1", "feature_2", "feature_3"]
        >>> vector = assemble_feature_vector(features, feature_order)
        >>> vector.shape
        (1, 3)
    """
    vector = np.array(
        [features.get(name, 0.0) for name in feature_order],
        dtype=np.float32
    ).reshape(1, -1)

    return vector

def score_transaction(
    tx: Dict[str, Any],
    model: Any,
    feature_order: list[str],
    threshold: float,
    model_version: str,
    feature_store: Optional[FeatureStore] = None
) -> Dict[str, Any]:
    """
    Score a transaction for fraud probability using ML model and feature store.

    This is the main prediction pipeline that:
    1. Validates transaction schema
    2. Extracts real-time features
    3. Enriches with historical features from feature store
    4. Makes fraud prediction with calibrated model
    5. Returns comprehensive prediction result

    Args:
        tx: Transaction dictionary with required fields:
            - customer_id (int): Customer identifier
            - terminal_id (int): Terminal/merchant identifier
            - amount (float): Transaction amount
            - timestamp (str): ISO format timestamp
            - transaction_id (str, optional): Transaction identifier
        model: Trained ML model with predict_proba method (e.g., XGBoost)
        feature_order: List of feature names in the order expected by model
        threshold: Decision threshold for fraud classification (typically 0.93)
        model_version: Model identifier for tracking (e.g., "XGBoost_v1.0")
        feature_store: FeatureStore instance (optional, will create if not provided)

    Returns:
        Dictionary containing:
            - transaction_id (str): Transaction identifier
            - customer_id (int): Customer ID
            - terminal_id (int): Terminal ID
            - amount (float): Transaction amount
            - fraud_probability (float): Probability of fraud [0.0-1.0]
            - is_fraud (bool): Binary fraud decision based on threshold
            - risk_level (str): Risk category (LOW/MEDIUM/HIGH/CRITICAL)
            - threshold (float): Decision threshold used
            - model_version (str): Model version identifier
            - timestamp (str): Transaction timestamp
            - features_used (dict): Key features that influenced prediction

    Raises:
        ValueError: If transaction schema is invalid

    Example:
        >>> tx = {
        ...     "transaction_id": "tx_123",
        ...     "customer_id": 100,
        ...     "terminal_id": 500,
        ...     "amount": 125.50,
        ...     "timestamp": "2024-01-15T14:30:00"
        ... }
        >>> result = score_transaction(tx, model, feature_order, 0.93, "XGBoost")
        >>> result["fraud_probability"]
        0.0234
        >>> result["is_fraud"]
        False
    """
    # 1. Validate transaction schema
    validate_schema(tx)

    # 2. Initialize feature store if not provided
    if feature_store is None:
        feature_store = FeatureStore()

    # 3. Extract real-time features
    realtime_features = extract_realtime_features(tx)

    # 4. Enrich with feature store (historical features)
    all_features = enrich_with_feature_store(tx, realtime_features, feature_store)

    # 5. Assemble feature vector in correct order
    vector = assemble_feature_vector(all_features, feature_order)

    # 6. Predict fraud probability
    fraud_prob = float(model.predict_proba(vector)[0, 1])
    is_fraud = fraud_prob >= threshold

    # 7. Categorize risk level
    if fraud_prob >= 0.9:
        risk_level = "CRITICAL"
    elif fraud_prob >= 0.7:
        risk_level = "HIGH"
    elif fraud_prob >= 0.5:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    # 8. Return comprehensive result
    return {
        "transaction_id": tx.get("transaction_id") or f"tx_{datetime.now().timestamp()}",
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
    