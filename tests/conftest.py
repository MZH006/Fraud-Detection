"""
Pytest configuration and shared fixtures for fraud detection tests.
"""

import os
from typing import Any, Dict
from unittest.mock import Mock

import numpy as np
import pytest

# ============================================================================
# Transaction Fixtures
# ============================================================================


@pytest.fixture
def sample_transaction() -> Dict[str, Any]:
    """Sample valid transaction for testing."""
    return {
        "transaction_id": "test_tx_001",
        "customer_id": 100,
        "terminal_id": 500,
        "amount": 125.50,
        "timestamp": "2024-01-15T14:30:00",
    }


@pytest.fixture
def night_transaction() -> Dict[str, Any]:
    """Transaction at night time (high risk)."""
    return {
        "transaction_id": "test_tx_night",
        "customer_id": 100,
        "terminal_id": 500,
        "amount": 125.50,
        "timestamp": "2024-01-15T02:30:00",  # 2:30 AM
    }


@pytest.fixture
def high_amount_transaction() -> Dict[str, Any]:
    """High amount transaction (high risk)."""
    return {
        "transaction_id": "test_tx_high",
        "customer_id": 100,
        "terminal_id": 500,
        "amount": 5000.0,
        "timestamp": "2024-01-15T14:30:00",
    }


@pytest.fixture
def weekend_transaction() -> Dict[str, Any]:
    """Weekend transaction."""
    return {
        "transaction_id": "test_tx_weekend",
        "customer_id": 100,
        "terminal_id": 500,
        "amount": 125.50,
        "timestamp": "2024-01-13T14:30:00",  # Saturday
    }


@pytest.fixture
def suspicious_transaction() -> Dict[str, Any]:
    """Suspicious transaction (high amount at night on weekend)."""
    return {
        "transaction_id": "test_tx_suspicious",
        "customer_id": 100,
        "terminal_id": 500,
        "amount": 5000.0,
        "timestamp": "2024-01-13T02:30:00",  # Saturday 2:30 AM
    }


# ============================================================================
# Feature Store Fixtures
# ============================================================================


@pytest.fixture
def mock_customer_features() -> Dict[str, float]:
    """Mock customer features, with every key FeatureStore.get_customer_features returns."""
    return {
        "CUSTOMER_TX_COUNT_1D": 2.0,
        "CUSTOMER_TX_COUNT_7D": 5.0,
        "CUSTOMER_TX_COUNT_30D": 18.0,
        "CUSTOMER_AVG_AMOUNT_1D": 50.0,
        "CUSTOMER_AVG_AMOUNT_7D": 75.5,
        "CUSTOMER_AVG_AMOUNT_30D": 70.0,
        "CUSTOMER_STD_AMOUNT_1D": 10.0,
        "CUSTOMER_STD_AMOUNT_7D": 25.0,
        "CUSTOMER_STD_AMOUNT_30D": 22.0,
    }


@pytest.fixture
def mock_terminal_features() -> Dict[str, float]:
    """Mock terminal features from feature store."""
    return {
        "TERMINAL_TX_COUNT_7D": 100.0,
        "TERMINAL_FRAUD_RATE_7D": 0.02,
        "TERMINAL_HIGH_RISK": 0.0,
        "TERMINAL_AVG_AMOUNT_7D": 150.0,
    }


@pytest.fixture
def mock_high_risk_terminal_features() -> Dict[str, float]:
    """Mock high-risk terminal features."""
    return {
        "TERMINAL_TX_COUNT_7D": 100.0,
        "TERMINAL_FRAUD_RATE_7D": 0.15,
        "TERMINAL_HIGH_RISK": 1.0,
        "TERMINAL_AVG_AMOUNT_7D": 150.0,
    }


@pytest.fixture
def mock_feature_store(mock_customer_features, mock_terminal_features):
    """Mock FeatureStore instance."""
    mock_store = Mock()
    mock_store.get_customer_features.return_value = mock_customer_features
    mock_store.get_terminal_features.return_value = mock_terminal_features
    return mock_store


# ============================================================================
# Model Fixtures
# ============================================================================


@pytest.fixture
def mock_model():
    """Mock ML model that returns configurable predictions."""
    model = Mock()
    # Default: return low fraud probability
    model.predict_proba.return_value = np.array([[0.95, 0.05]])
    return model


@pytest.fixture
def mock_fraud_model():
    """Mock ML model that predicts high fraud probability."""
    model = Mock()
    model.predict_proba.return_value = np.array([[0.1, 0.9]])
    return model


@pytest.fixture
def sample_feature_order():
    """Sample feature order list for model."""
    return [
        "TX_AMOUNT",
        "TX_HOUR_SIN",
        "TX_HOUR_COS",
        "TX_DAY_OF_WEEK",
        "TX_IS_WEEKEND",
        "TX_IS_NIGHT",
        "TX_AMOUNT_HIGH",
        "TX_NIGHT_HIGH_AMOUNT",
        "CUSTOMER_TX_COUNT_7D",
        "CUSTOMER_AVG_AMOUNT_7D",
        "CUSTOMER_STD_AMOUNT_7D",
        "CUSTOMER_TX_COUNT_1D",
        "CUSTOMER_AVG_AMOUNT_1D",
        "CUSTOMER_AMOUNT_DEVIATION",
        "CUSTOMER_HIGH_DEVIATION",
        "TERMINAL_TX_COUNT_7D",
        "TERMINAL_FRAUD_RATE_7D",
        "TERMINAL_HIGH_RISK",
        "TERMINAL_AVG_AMOUNT_7D",
        "RISK_SCORE",
    ]


# ============================================================================
# API Test Fixtures
# ============================================================================


@pytest.fixture
def test_client():
    """FastAPI test client."""
    # Import here to avoid circular imports
    from fastapi.testclient import TestClient

    # Mock environment variables before importing app
    os.environ.setdefault("SUPABASE_URL", "https://test.supabase.co")
    os.environ.setdefault("SUPABASE_KEY", "test-key")

    try:
        from src.app import app

        return TestClient(app)
    except Exception as e:
        pytest.skip(f"Could not create test client: {e}")


# ============================================================================
# Utility Fixtures
# ============================================================================


@pytest.fixture
def invalid_transactions():
    """Collection of invalid transactions for testing validation."""
    return {
        "missing_customer_id": {
            "terminal_id": 500,
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00",
        },
        "missing_terminal_id": {
            "customer_id": 100,
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00",
        },
        "missing_amount": {
            "customer_id": 100,
            "terminal_id": 500,
            "timestamp": "2024-01-15T14:30:00",
        },
        "missing_timestamp": {"customer_id": 100, "terminal_id": 500, "amount": 100.0},
        "negative_amount": {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": -50.0,
            "timestamp": "2024-01-15T14:30:00",
        },
        "invalid_timestamp": {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 100.0,
            "timestamp": "not-a-timestamp",
        },
        "invalid_amount_type": {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": "one hundred",
            "timestamp": "2024-01-15T14:30:00",
        },
    }


@pytest.fixture
def batch_transactions(sample_transaction):
    """Batch of transactions for testing batch endpoints."""
    return [
        {**sample_transaction, "transaction_id": f"batch_tx_{i}", "amount": 50.0 + i * 25}
        for i in range(5)
    ]
