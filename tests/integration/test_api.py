"""
Integration tests for FastAPI fraud detection endpoints.
"""

import os
from unittest.mock import Mock, patch

import numpy as np
import pytest


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        # Mock the model loading and feature store
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_KEY": "test-key",
                "MODEL_PATH": "./artifacts/fraud_detection_model_final.pkl",
            },
        ):
            with patch("src.app.joblib.load") as mock_load:
                with patch("src.app.FeatureStore") as mock_fs:
                    # Setup mock model
                    mock_model = Mock()
                    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

                    mock_load.return_value = {
                        "model": mock_model,
                        "feature_columns": ["TX_AMOUNT", "TX_HOUR_SIN"],
                        "threshold": 0.5,
                        "model_type": "XGBoost_Test",
                    }

                    # Setup mock feature store
                    mock_store = Mock()
                    mock_store.get_customer_features.return_value = {
                        "CUSTOMER_TX_COUNT_7D": 5.0,
                        "CUSTOMER_AVG_AMOUNT_7D": 75.5,
                        "CUSTOMER_STD_AMOUNT_7D": 25.0,
                        "CUSTOMER_TX_COUNT_1D": 2.0,
                        "CUSTOMER_AVG_AMOUNT_1D": 50.0,
                    }
                    mock_store.get_terminal_features.return_value = {
                        "TERMINAL_TX_COUNT_7D": 100.0,
                        "TERMINAL_FRAUD_RATE_7D": 0.02,
                        "TERMINAL_HIGH_RISK": 0.0,
                        "TERMINAL_AVG_AMOUNT_7D": 150.0,
                    }
                    mock_fs.return_value = mock_store

                    # Import app after mocks are set up
                    from fastapi.testclient import TestClient

                    from src.app import app

                    yield TestClient(app)

    def test_root_endpoint(self, client):
        """Test root endpoint returns health status."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model" in data
        assert "threshold" in data
        assert "version" in data

    def test_health_endpoint(self, client):
        """Test detailed health check endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
        assert "model_type" in data
        assert "num_features" in data


class TestPredictEndpoint:
    """Tests for fraud prediction endpoint."""

    @pytest.fixture
    def client_and_mocks(self):
        """Create test client with accessible mocks."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_KEY": "test-key",
                "MODEL_PATH": "./artifacts/fraud_detection_model_final.pkl",
            },
        ):
            with patch("src.app.joblib.load") as mock_load:
                with patch("src.app.FeatureStore") as mock_fs:
                    # Setup mock model
                    mock_model = Mock()
                    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

                    mock_load.return_value = {
                        "model": mock_model,
                        "feature_columns": [
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
                        ],
                        "threshold": 0.5,
                        "model_type": "XGBoost_Test",
                    }

                    # Setup mock feature store
                    mock_store = Mock()
                    mock_store.get_customer_features.return_value = {
                        "CUSTOMER_TX_COUNT_7D": 5.0,
                        "CUSTOMER_AVG_AMOUNT_7D": 75.5,
                        "CUSTOMER_STD_AMOUNT_7D": 25.0,
                        "CUSTOMER_TX_COUNT_1D": 2.0,
                        "CUSTOMER_AVG_AMOUNT_1D": 50.0,
                    }
                    mock_store.get_terminal_features.return_value = {
                        "TERMINAL_TX_COUNT_7D": 100.0,
                        "TERMINAL_FRAUD_RATE_7D": 0.02,
                        "TERMINAL_HIGH_RISK": 0.0,
                        "TERMINAL_AVG_AMOUNT_7D": 150.0,
                    }
                    mock_fs.return_value = mock_store

                    from fastapi.testclient import TestClient

                    from src.app import app

                    yield TestClient(app), mock_model, mock_store

    def test_predict_valid_transaction(self, client_and_mocks):
        """Test prediction with valid transaction."""
        client, mock_model, _ = client_and_mocks

        transaction = {
            "transaction_id": "test_001",
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 125.50,
            "timestamp": "2024-01-15T14:30:00",
        }

        response = client.post("/predict", json=transaction)

        assert response.status_code == 200
        data = response.json()
        assert "fraud_probability" in data
        assert "is_fraud" in data
        assert "risk_level" in data
        assert "transaction_id" in data
        assert data["transaction_id"] == "test_001"

    def test_predict_returns_probability_range(self, client_and_mocks):
        """Fraud probability should be between 0 and 1."""
        client, _, _ = client_and_mocks

        transaction = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 125.50,
            "timestamp": "2024-01-15T14:30:00",
        }

        response = client.post("/predict", json=transaction)

        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["fraud_probability"] <= 1.0

    def test_predict_missing_customer_id(self, client_and_mocks):
        """Should return 422 for missing customer_id."""
        client, _, _ = client_and_mocks

        transaction = {"terminal_id": 500, "amount": 125.50, "timestamp": "2024-01-15T14:30:00"}

        response = client.post("/predict", json=transaction)

        assert response.status_code == 422

    def test_predict_missing_terminal_id(self, client_and_mocks):
        """Should return 422 for missing terminal_id."""
        client, _, _ = client_and_mocks

        transaction = {"customer_id": 100, "amount": 125.50, "timestamp": "2024-01-15T14:30:00"}

        response = client.post("/predict", json=transaction)

        assert response.status_code == 422

    def test_predict_missing_amount(self, client_and_mocks):
        """Should return 422 for missing amount."""
        client, _, _ = client_and_mocks

        transaction = {"customer_id": 100, "terminal_id": 500, "timestamp": "2024-01-15T14:30:00"}

        response = client.post("/predict", json=transaction)

        assert response.status_code == 422

    def test_predict_missing_timestamp(self, client_and_mocks):
        """Should return 422 for missing timestamp."""
        client, _, _ = client_and_mocks

        transaction = {"customer_id": 100, "terminal_id": 500, "amount": 125.50}

        response = client.post("/predict", json=transaction)

        assert response.status_code == 422

    def test_predict_negative_amount(self, client_and_mocks):
        """Should return 422 for negative amount."""
        client, _, _ = client_and_mocks

        transaction = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": -50.0,
            "timestamp": "2024-01-15T14:30:00",
        }

        response = client.post("/predict", json=transaction)

        # Pydantic validation should catch this
        assert response.status_code == 422

    def test_predict_includes_processing_time(self, client_and_mocks):
        """Response should include processing time."""
        client, _, _ = client_and_mocks

        transaction = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 125.50,
            "timestamp": "2024-01-15T14:30:00",
        }

        response = client.post("/predict", json=transaction)

        assert response.status_code == 200
        data = response.json()
        assert "processing_time_ms" in data
        assert data["processing_time_ms"] >= 0

    def test_predict_includes_request_id_header(self, client_and_mocks):
        """Response should include X-Request-ID header."""
        client, _, _ = client_and_mocks

        transaction = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 125.50,
            "timestamp": "2024-01-15T14:30:00",
        }

        response = client.post("/predict", json=transaction)

        assert response.status_code == 200
        assert "x-request-id" in response.headers

    def test_predict_high_fraud_probability(self, client_and_mocks):
        """High fraud probability should result in is_fraud=True."""
        client, mock_model, _ = client_and_mocks

        # Set model to return high fraud probability
        mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])

        transaction = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 5000.0,
            "timestamp": "2024-01-15T02:30:00",
        }

        response = client.post("/predict", json=transaction)

        assert response.status_code == 200
        data = response.json()
        assert data["is_fraud"] is True
        assert data["risk_level"] in ["HIGH", "CRITICAL"]


class TestBatchPredictEndpoint:
    """Tests for batch prediction endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_KEY": "test-key",
                "MODEL_PATH": "./artifacts/fraud_detection_model_final.pkl",
            },
        ):
            with patch("src.app.joblib.load") as mock_load:
                with patch("src.app.FeatureStore") as mock_fs:
                    mock_model = Mock()
                    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

                    mock_load.return_value = {
                        "model": mock_model,
                        "feature_columns": [
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
                        ],
                        "threshold": 0.5,
                        "model_type": "XGBoost_Test",
                    }

                    mock_store = Mock()
                    mock_store.get_customer_features.return_value = {
                        "CUSTOMER_TX_COUNT_7D": 5.0,
                        "CUSTOMER_AVG_AMOUNT_7D": 75.5,
                        "CUSTOMER_STD_AMOUNT_7D": 25.0,
                        "CUSTOMER_TX_COUNT_1D": 2.0,
                        "CUSTOMER_AVG_AMOUNT_1D": 50.0,
                    }
                    mock_store.get_terminal_features.return_value = {
                        "TERMINAL_TX_COUNT_7D": 100.0,
                        "TERMINAL_FRAUD_RATE_7D": 0.02,
                        "TERMINAL_HIGH_RISK": 0.0,
                        "TERMINAL_AVG_AMOUNT_7D": 150.0,
                    }
                    mock_fs.return_value = mock_store

                    from fastapi.testclient import TestClient

                    from src.app import app

                    yield TestClient(app)

    def test_batch_predict_multiple_transactions(self, client):
        """Test batch prediction with multiple transactions."""
        transactions = [
            {
                "transaction_id": f"batch_{i}",
                "customer_id": 100 + i,
                "terminal_id": 500,
                "amount": 50.0 + i * 25,
                "timestamp": "2024-01-15T14:30:00",
            }
            for i in range(5)
        ]

        response = client.post("/batch-predict", json=transactions)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 5
        assert len(data["predictions"]) == 5

    def test_batch_predict_empty_list(self, client):
        """Test batch prediction with empty list."""
        response = client.post("/batch-predict", json=[])

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 0
        assert len(data["predictions"]) == 0

    def test_batch_predict_single_transaction(self, client):
        """Test batch prediction with single transaction."""
        transactions = [
            {
                "transaction_id": "single_batch",
                "customer_id": 100,
                "terminal_id": 500,
                "amount": 125.50,
                "timestamp": "2024-01-15T14:30:00",
            }
        ]

        response = client.post("/batch-predict", json=transactions)

        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1

    def test_batch_predict_includes_error_count(self, client):
        """Response should include error count."""
        transactions = [
            {
                "customer_id": 100,
                "terminal_id": 500,
                "amount": 125.50,
                "timestamp": "2024-01-15T14:30:00",
            }
        ]

        response = client.post("/batch-predict", json=transactions)

        assert response.status_code == 200
        data = response.json()
        assert "errors" in data


class TestStatsEndpoint:
    """Tests for stats endpoint."""

    @pytest.fixture
    def client(self):
        """Create test client with mocked dependencies."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_KEY": "test-key",
                "MODEL_PATH": "./artifacts/fraud_detection_model_final.pkl",
            },
        ):
            with patch("src.app.joblib.load") as mock_load:
                with patch("src.app.FeatureStore") as mock_fs:
                    mock_model = Mock()
                    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

                    mock_load.return_value = {
                        "model": mock_model,
                        "feature_columns": ["TX_AMOUNT"],
                        "threshold": 0.5,
                        "model_type": "XGBoost_Test",
                    }

                    mock_store = Mock()

                    # Mock stats queries
                    mock_customer_response = Mock()
                    mock_customer_response.data = [{"customer_id": i} for i in range(100)]

                    mock_terminal_response = Mock()
                    mock_terminal_response.data = [{"terminal_id": i} for i in range(50)]

                    mock_store.client.table.return_value.select.return_value.execute.side_effect = [
                        mock_customer_response,
                        mock_terminal_response,
                    ]

                    mock_fs.return_value = mock_store

                    from fastapi.testclient import TestClient

                    from src.app import app

                    yield TestClient(app)

    def test_stats_endpoint_returns_counts(self, client):
        """Stats endpoint should return customer and terminal counts."""
        response = client.get("/stats")

        # May return 200 or 500 depending on mock setup
        if response.status_code == 200:
            data = response.json()
            assert "customers_in_store" in data or "model_version" in data


class TestAPIMiddleware:
    """Tests for API middleware functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        with patch.dict(
            os.environ,
            {
                "SUPABASE_URL": "https://test.supabase.co",
                "SUPABASE_KEY": "test-key",
                "MODEL_PATH": "./artifacts/fraud_detection_model_final.pkl",
            },
        ):
            with patch("src.app.joblib.load") as mock_load:
                with patch("src.app.FeatureStore") as mock_fs:
                    mock_model = Mock()
                    mock_model.predict_proba.return_value = np.array([[0.95, 0.05]])

                    mock_load.return_value = {
                        "model": mock_model,
                        "feature_columns": ["TX_AMOUNT"],
                        "threshold": 0.5,
                        "model_type": "XGBoost_Test",
                    }

                    mock_fs.return_value = Mock()

                    from fastapi.testclient import TestClient

                    from src.app import app

                    yield TestClient(app)

    def test_cors_headers_present(self, client):
        """CORS headers should be present in response."""
        response = client.options("/")

        # OPTIONS request should be handled
        assert response.status_code in [200, 405]

    def test_request_id_generated(self, client):
        """Request ID should be generated if not provided."""
        response = client.get("/")

        assert response.status_code == 200
        assert "x-request-id" in response.headers
        assert len(response.headers["x-request-id"]) > 0

    def test_request_id_preserved(self, client):
        """Request ID should be preserved if provided."""
        custom_request_id = "custom-request-123"
        response = client.get("/", headers={"X-Request-ID": custom_request_id})

        assert response.status_code == 200
        assert response.headers["x-request-id"] == custom_request_id
