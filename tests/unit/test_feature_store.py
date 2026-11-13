"""
Unit tests for feature_store/store.py - Feature store operations.
"""

import os
from unittest.mock import Mock, patch

import pytest


class TestFeatureStoreInit:
    """Tests for FeatureStore initialization."""

    def test_missing_credentials_raises(self):
        """Should raise ValueError if credentials are missing."""
        with patch.dict(os.environ, {"SUPABASE_URL": "", "SUPABASE_KEY": ""}, clear=True):
            # Need to reimport to pick up new env vars
            import importlib

            import src.feature_store.store as store_module

            importlib.reload(store_module)

            with pytest.raises(ValueError, match="SUPABASE_URL and SUPABASE_KEY must be set"):
                store_module.FeatureStore()

    @patch("src.feature_store.store.create_client")
    def test_successful_connection(self, mock_create_client):
        """Should connect successfully with valid credentials."""
        with patch.dict(
            os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}
        ):
            mock_client = Mock()
            mock_create_client.return_value = mock_client

            from src.feature_store.store import FeatureStore

            store = FeatureStore()

            mock_create_client.assert_called_once()
            assert store.client == mock_client


class TestGetCustomerFeatures:
    """Tests for get_customer_features method."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client"""
        with patch("src.feature_store.store.create_client") as mock:
            with patch.dict(
                os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}
            ):
                yield mock

    def test_get_customer_features_success(self, mock_supabase_client):
        """Test successful customer feature retrieval"""
        # Setup mock
        mock_response = Mock()
        mock_response.data = [
            {
                "customer_id": 100,
                "tx_count_7d": 5,
                "avg_amount_7d": 75.5,
                "std_amount_7d": 20.0,
                "tx_count_1d": 2,
                "avg_amount_1d": 50.0,
            }
        ]

        mock_client = Mock()
        mock_client.table().select().eq().execute.return_value = mock_response
        mock_supabase_client.return_value = mock_client

        from src.feature_store.store import FeatureStore

        store = FeatureStore()
        features = store.get_customer_features(100)

        assert features["CUSTOMER_TX_COUNT_7D"] == 5
        assert features["CUSTOMER_AVG_AMOUNT_7D"] == 75.5

    def test_get_customer_features_not_found(self, mock_supabase_client):
        """Test customer not found returns defaults"""
        mock_response = Mock()
        mock_response.data = []

        mock_client = Mock()
        mock_client.table().select().eq().execute.return_value = mock_response
        mock_supabase_client.return_value = mock_client

        from src.feature_store.store import FeatureStore

        store = FeatureStore()
        features = store.get_customer_features(999999)

        # Should return default values
        assert features["CUSTOMER_TX_COUNT_7D"] == 0
        assert features["CUSTOMER_AVG_AMOUNT_7D"] == 50.0

    def test_get_customer_features_handles_error(self, mock_supabase_client):
        """Test graceful error handling"""
        mock_client = Mock()
        mock_client.table().select().eq().execute.side_effect = Exception("DB Error")
        mock_supabase_client.return_value = mock_client

        from src.feature_store.store import FeatureStore

        store = FeatureStore()

        # Should return defaults on error, not raise
        features = store.get_customer_features(100)
        assert "CUSTOMER_TX_COUNT_7D" in features


class TestGetTerminalFeatures:
    """Tests for get_terminal_features method."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client"""
        with patch("src.feature_store.store.create_client") as mock:
            with patch.dict(
                os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}
            ):
                yield mock

    def test_get_terminal_features_success(self, mock_supabase_client):
        """Test successful terminal feature retrieval"""
        mock_response = Mock()
        mock_response.data = [
            {
                "terminal_id": 500,
                "tx_count_7d": 100,
                "fraud_rate_7d": 0.02,
                "high_risk": False,
                "avg_amount_7d": 150.0,
            }
        ]

        mock_client = Mock()
        mock_client.table().select().eq().execute.return_value = mock_response
        mock_supabase_client.return_value = mock_client

        from src.feature_store.store import FeatureStore

        store = FeatureStore()
        features = store.get_terminal_features(500)

        assert "TERMINAL_TX_COUNT_7D" in features
        assert "TERMINAL_FRAUD_RATE_7D" in features

    def test_get_terminal_features_high_risk(self, mock_supabase_client):
        """Test terminal with high fraud rate flagged as high risk"""
        mock_response = Mock()
        mock_response.data = [
            {
                "terminal_id": 500,
                "tx_count_7d": 100,
                "fraud_rate_7d": 0.08,  # 8% fraud rate
                "high_risk": True,
                "avg_amount_7d": 150.0,
            }
        ]

        mock_client = Mock()
        mock_client.table().select().eq().execute.return_value = mock_response
        mock_supabase_client.return_value = mock_client

        from src.feature_store.store import FeatureStore

        store = FeatureStore()
        features = store.get_terminal_features(500)

        assert features["TERMINAL_HIGH_RISK"] == 1
        assert features["TERMINAL_FRAUD_RATE_7D"] == 0.08

    def test_get_terminal_features_not_found(self, mock_supabase_client):
        """Test terminal not found returns defaults"""
        mock_response = Mock()
        mock_response.data = []

        mock_client = Mock()
        mock_client.table().select().eq().execute.return_value = mock_response
        mock_supabase_client.return_value = mock_client

        from src.feature_store.store import FeatureStore

        store = FeatureStore()
        features = store.get_terminal_features(99999)

        # Should return default values
        assert features["TERMINAL_TX_COUNT_7D"] == 0
        assert features["TERMINAL_FRAUD_RATE_7D"] == 0.0
        assert features["TERMINAL_HIGH_RISK"] == 0.0


class TestLogTransaction:
    """Tests for log_transaction method."""

    @pytest.fixture
    def mock_supabase_client(self):
        """Mock Supabase client"""
        with patch("src.feature_store.store.create_client") as mock:
            with patch.dict(
                os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}
            ):
                yield mock

    def test_logs_transaction_successfully(self, mock_supabase_client):
        """Test successful transaction logging"""
        mock_client = Mock()
        mock_supabase_client.return_value = mock_client

        from src.feature_store.store import FeatureStore

        store = FeatureStore()

        transaction = {
            "transaction_id": "tx_001",
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 125.50,
            "timestamp": "2024-01-15T14:30:00",
            "is_fraud": False,
            "fraud_probability": 0.05,
        }

        # Should not raise
        store.log_transaction(transaction)

        # Verify insert was called
        mock_client.table.return_value.insert.assert_called()

    def test_handles_logging_error(self, mock_supabase_client):
        """Test graceful handling of logging errors"""
        mock_client = Mock()
        mock_client.table().insert().execute.side_effect = Exception("Insert Error")
        mock_supabase_client.return_value = mock_client

        from src.feature_store.store import FeatureStore

        store = FeatureStore()

        transaction = {
            "transaction_id": "tx_001",
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 125.50,
            "timestamp": "2024-01-15T14:30:00",
            "is_fraud": False,
            "fraud_probability": 0.05,
        }

        # Should not raise - errors should be caught and logged
        store.log_transaction(transaction)


class TestFeatureStoreDefaults:
    """Tests for default feature values."""

    @pytest.fixture
    def mock_store_empty_responses(self):
        """Create store with empty responses for all queries."""
        with patch("src.feature_store.store.create_client") as mock_create:
            with patch.dict(
                os.environ, {"SUPABASE_URL": "https://test.supabase.co", "SUPABASE_KEY": "test-key"}
            ):
                mock_client = Mock()
                mock_response = Mock()
                mock_response.data = []
                mock_client.table().select().eq().execute.return_value = mock_response
                mock_create.return_value = mock_client

                from src.feature_store.store import FeatureStore

                store = FeatureStore()

                yield store

    def test_customer_defaults_are_reasonable(self, mock_store_empty_responses):
        """Customer default values should be reasonable."""
        features = mock_store_empty_responses.get_customer_features(99999)

        # Transaction count should be 0 for new customer
        assert features["CUSTOMER_TX_COUNT_7D"] == 0
        assert features["CUSTOMER_TX_COUNT_1D"] == 0

        # Average amount should have reasonable default
        assert features["CUSTOMER_AVG_AMOUNT_7D"] > 0
        assert features["CUSTOMER_AVG_AMOUNT_1D"] > 0

        # Standard deviation should have reasonable default
        assert features["CUSTOMER_STD_AMOUNT_7D"] > 0

    def test_terminal_defaults_are_reasonable(self, mock_store_empty_responses):
        """Terminal default values should be reasonable."""
        features = mock_store_empty_responses.get_terminal_features(99999)

        # Transaction count should be 0 for new terminal
        assert features["TERMINAL_TX_COUNT_7D"] == 0

        # Fraud rate should be 0 for new terminal
        assert features["TERMINAL_FRAUD_RATE_7D"] == 0.0

        # Should not be flagged as high risk by default
        assert features["TERMINAL_HIGH_RISK"] == 0.0
