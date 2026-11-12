"""
Unit tests for model_utils.py - Feature extraction and scoring functions.
"""

import pytest
import math
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model_utils import (
    validate_schema,
    extract_realtime_features,
    enrich_with_feature_store,
    assemble_feature_vector,
    score_transaction,
    REQUIRED_FIELDS
)


class TestValidateSchema:
    """Tests for transaction schema validation."""

    def test_valid_transaction_passes(self, sample_transaction):
        """Valid transaction should pass validation without error."""
        # Should not raise
        validate_schema(sample_transaction)

    def test_valid_transaction_without_optional_fields(self):
        """Transaction without optional fields should pass."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        validate_schema(tx)  # Should not raise

    def test_missing_customer_id_raises(self):
        """Missing customer_id should raise ValueError."""
        tx = {
            "terminal_id": 500,
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_schema(tx)

    def test_missing_terminal_id_raises(self):
        """Missing terminal_id should raise ValueError."""
        tx = {
            "customer_id": 100,
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_schema(tx)

    def test_missing_amount_raises(self):
        """Missing amount should raise ValueError."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "timestamp": "2024-01-15T14:30:00"
        }
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_schema(tx)

    def test_missing_timestamp_raises(self):
        """Missing timestamp should raise ValueError."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 100.0
        }
        with pytest.raises(ValueError, match="Missing required fields"):
            validate_schema(tx)

    def test_negative_amount_raises(self):
        """Negative amount should raise ValueError."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": -50.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        with pytest.raises(ValueError, match="Amount must be non-negative"):
            validate_schema(tx)

    def test_zero_amount_passes(self):
        """Zero amount should be valid."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 0.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        validate_schema(tx)  # Should not raise

    def test_string_amount_raises(self):
        """Non-numeric amount should raise ValueError."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": "one hundred",
            "timestamp": "2024-01-15T14:30:00"
        }
        with pytest.raises(ValueError, match="Amount must be numeric"):
            validate_schema(tx)

    def test_integer_amount_passes(self):
        """Integer amount should be valid."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 100,  # int, not float
            "timestamp": "2024-01-15T14:30:00"
        }
        validate_schema(tx)  # Should not raise

    def test_string_customer_id_passes(self):
        """String customer_id should be valid."""
        tx = {
            "customer_id": "cust_100",
            "terminal_id": 500,
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        validate_schema(tx)  # Should not raise


class TestExtractRealtimeFeatures:
    """Tests for real-time feature extraction."""

    def test_basic_feature_extraction(self, sample_transaction):
        """Test basic feature extraction from valid transaction."""
        features = extract_realtime_features(sample_transaction)

        assert "TX_AMOUNT" in features
        assert features["TX_AMOUNT"] == 125.50
        assert "TX_HOUR_SIN" in features
        assert "TX_HOUR_COS" in features
        assert "TX_DAY_OF_WEEK" in features
        assert "TX_IS_WEEKEND" in features
        assert "TX_IS_NIGHT" in features
        assert "TX_AMOUNT_HIGH" in features
        assert "TX_NIGHT_HIGH_AMOUNT" in features

    def test_cyclical_hour_encoding(self, sample_transaction):
        """Test that hour encoding is cyclical (sin/cos)."""
        features = extract_realtime_features(sample_transaction)

        # Hour 14:30 -> hour = 14
        hour = 14
        expected_sin = math.sin(2 * math.pi * hour / 24)
        expected_cos = math.cos(2 * math.pi * hour / 24)

        assert abs(features["TX_HOUR_SIN"] - expected_sin) < 0.0001
        assert abs(features["TX_HOUR_COS"] - expected_cos) < 0.0001

    def test_weekend_detection_saturday(self):
        """Saturday should be detected as weekend."""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-13T14:30:00"  # Saturday
        }
        features = extract_realtime_features(tx)
        assert features["TX_IS_WEEKEND"] == 1.0

    def test_weekend_detection_sunday(self):
        """Sunday should be detected as weekend."""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-14T14:30:00"  # Sunday
        }
        features = extract_realtime_features(tx)
        assert features["TX_IS_WEEKEND"] == 1.0

    def test_weekday_detection(self):
        """Monday should not be detected as weekend."""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"  # Monday
        }
        features = extract_realtime_features(tx)
        assert features["TX_IS_WEEKEND"] == 0.0

    def test_night_detection_2am(self):
        """2 AM should be detected as night."""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-15T02:30:00"
        }
        features = extract_realtime_features(tx)
        assert features["TX_IS_NIGHT"] == 1.0

    def test_night_detection_11pm(self):
        """11 PM should be detected as night."""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-15T23:30:00"
        }
        features = extract_realtime_features(tx)
        assert features["TX_IS_NIGHT"] == 1.0

    def test_day_detection_2pm(self):
        """2 PM should not be detected as night."""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        features = extract_realtime_features(tx)
        assert features["TX_IS_NIGHT"] == 0.0

    def test_high_amount_detection(self):
        """Amount > $200 should be flagged as high."""
        tx = {
            "amount": 250.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        features = extract_realtime_features(tx)
        assert features["TX_AMOUNT_HIGH"] == 1.0

    def test_low_amount_not_flagged(self):
        """Amount <= $200 should not be flagged as high."""
        tx = {
            "amount": 150.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        features = extract_realtime_features(tx)
        assert features["TX_AMOUNT_HIGH"] == 0.0

    def test_boundary_amount_200(self):
        """Amount exactly $200 should not be flagged (> 200 required)."""
        tx = {
            "amount": 200.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        features = extract_realtime_features(tx)
        assert features["TX_AMOUNT_HIGH"] == 0.0

    def test_night_high_amount_interaction(self):
        """Night + high amount should set interaction feature."""
        tx = {
            "amount": 300.0,
            "timestamp": "2024-01-15T02:30:00"  # Night + high amount
        }
        features = extract_realtime_features(tx)
        assert features["TX_NIGHT_HIGH_AMOUNT"] == 1.0

    def test_night_low_amount_no_interaction(self):
        """Night + low amount should not set interaction feature."""
        tx = {
            "amount": 50.0,
            "timestamp": "2024-01-15T02:30:00"  # Night + low amount
        }
        features = extract_realtime_features(tx)
        assert features["TX_NIGHT_HIGH_AMOUNT"] == 0.0

    def test_day_high_amount_no_interaction(self):
        """Day + high amount should not set interaction feature."""
        tx = {
            "amount": 300.0,
            "timestamp": "2024-01-15T14:30:00"  # Day + high amount
        }
        features = extract_realtime_features(tx)
        assert features["TX_NIGHT_HIGH_AMOUNT"] == 0.0

    def test_invalid_timestamp_raises(self):
        """Invalid timestamp should raise ValueError."""
        tx = {
            "amount": 100.0,
            "timestamp": "not-a-timestamp"
        }
        with pytest.raises(ValueError, match="Invalid timestamp format"):
            extract_realtime_features(tx)

    def test_timezone_z_suffix(self):
        """Timestamp with Z suffix should be parsed correctly."""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00Z"
        }
        features = extract_realtime_features(tx)
        assert features["TX_AMOUNT"] == 100.0

    def test_day_of_week_monday(self):
        """Monday should have day_of_week = 0."""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"  # Monday
        }
        features = extract_realtime_features(tx)
        assert features["TX_DAY_OF_WEEK"] == 0.0

    def test_day_of_week_sunday(self):
        """Sunday should have day_of_week = 6."""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-14T14:30:00"  # Sunday
        }
        features = extract_realtime_features(tx)
        assert features["TX_DAY_OF_WEEK"] == 6.0

    def test_all_features_are_float(self, sample_transaction):
        """All extracted features should be float type."""
        features = extract_realtime_features(sample_transaction)
        for key, value in features.items():
            assert isinstance(value, float), f"{key} is not float: {type(value)}"


class TestEnrichWithFeatureStore:
    """Tests for feature store enrichment."""

    def test_combines_all_features(
        self, sample_transaction, mock_feature_store, mock_customer_features, mock_terminal_features
    ):
        """Should combine realtime, customer, and terminal features."""
        realtime = extract_realtime_features(sample_transaction)
        all_features = enrich_with_feature_store(sample_transaction, realtime, mock_feature_store)

        # Check realtime features are present
        assert "TX_AMOUNT" in all_features

        # Check customer features are present
        assert "CUSTOMER_AVG_AMOUNT_7D" in all_features

        # Check terminal features are present
        assert "TERMINAL_FRAUD_RATE_7D" in all_features

        # Check derived features are computed
        assert "CUSTOMER_AMOUNT_DEVIATION" in all_features
        assert "CUSTOMER_HIGH_DEVIATION" in all_features
        assert "RISK_SCORE" in all_features

    def test_amount_deviation_calculation(
        self, sample_transaction, mock_feature_store, mock_customer_features
    ):
        """Amount deviation should be calculated correctly."""
        # Transaction amount is 125.50
        # Customer avg is 75.5, std is 25.0
        # Deviation = (125.50 - 75.5) / 25.0 = 2.0

        realtime = extract_realtime_features(sample_transaction)
        all_features = enrich_with_feature_store(sample_transaction, realtime, mock_feature_store)

        expected_deviation = (125.50 - 75.5) / 25.0
        assert abs(all_features["CUSTOMER_AMOUNT_DEVIATION"] - expected_deviation) < 0.0001

    def test_high_deviation_flag(self, mock_feature_store):
        """High deviation (>2) should be flagged."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 200.0,  # Much higher than avg (75.5)
            "timestamp": "2024-01-15T14:30:00"
        }
        realtime = extract_realtime_features(tx)
        all_features = enrich_with_feature_store(tx, realtime, mock_feature_store)

        # (200 - 75.5) / 25 = 4.98 > 2
        assert all_features["CUSTOMER_HIGH_DEVIATION"] == 1.0

    def test_low_deviation_not_flagged(self, sample_transaction, mock_feature_store):
        """Normal deviation should not be flagged."""
        tx = {**sample_transaction, "amount": 80.0}  # Close to avg (75.5)
        realtime = extract_realtime_features(tx)
        all_features = enrich_with_feature_store(tx, realtime, mock_feature_store)

        # (80 - 75.5) / 25 = 0.18 < 2
        assert all_features["CUSTOMER_HIGH_DEVIATION"] == 0.0

    def test_deviation_clipping(self, mock_feature_store):
        """Extreme deviations should be clipped to [-10, 10]."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 10000.0,  # Extreme amount
            "timestamp": "2024-01-15T14:30:00"
        }
        realtime = extract_realtime_features(tx)
        all_features = enrich_with_feature_store(tx, realtime, mock_feature_store)

        # (10000 - 75.5) / 25 = 396.98 -> clipped to 10
        assert all_features["CUSTOMER_AMOUNT_DEVIATION"] == 10.0

    def test_zero_std_deviation(self, sample_transaction, mock_customer_features, mock_terminal_features):
        """Zero std deviation should result in 0 deviation."""
        mock_store = Mock()
        zero_std_features = {**mock_customer_features, 'CUSTOMER_STD_AMOUNT_7D': 0.0}
        mock_store.get_customer_features.return_value = zero_std_features
        mock_store.get_terminal_features.return_value = mock_terminal_features

        realtime = extract_realtime_features(sample_transaction)
        all_features = enrich_with_feature_store(sample_transaction, realtime, mock_store)

        assert all_features["CUSTOMER_AMOUNT_DEVIATION"] == 0.0

    def test_risk_score_calculation(
        self, sample_transaction, mock_feature_store
    ):
        """Risk score should be weighted sum of risk indicators."""
        realtime = extract_realtime_features(sample_transaction)
        all_features = enrich_with_feature_store(sample_transaction, realtime, mock_feature_store)

        # Risk = TERMINAL_HIGH_RISK * 2 + HIGH_DEVIATION * 3 + TX_NIGHT_HIGH_AMOUNT
        # For sample transaction: 0 * 2 + 0 * 3 + 0 = 0
        assert all_features["RISK_SCORE"] == 0.0

    def test_risk_score_high_risk_terminal(
        self, sample_transaction, mock_customer_features, mock_high_risk_terminal_features
    ):
        """High risk terminal should increase risk score."""
        mock_store = Mock()
        mock_store.get_customer_features.return_value = mock_customer_features
        mock_store.get_terminal_features.return_value = mock_high_risk_terminal_features

        realtime = extract_realtime_features(sample_transaction)
        all_features = enrich_with_feature_store(sample_transaction, realtime, mock_store)

        # TERMINAL_HIGH_RISK = 1, so risk_score >= 2
        assert all_features["RISK_SCORE"] >= 2.0


class TestAssembleFeatureVector:
    """Tests for feature vector assembly."""

    def test_correct_ordering(self):
        """Features should be assembled in specified order."""
        features = {"a": 1.0, "b": 2.0, "c": 3.0}
        feature_order = ["c", "a", "b"]

        vector = assemble_feature_vector(features, feature_order)

        assert vector.shape == (1, 3)
        assert vector[0, 0] == 3.0  # c
        assert vector[0, 1] == 1.0  # a
        assert vector[0, 2] == 2.0  # b

    def test_missing_feature_defaults_to_zero(self):
        """Missing features should default to 0."""
        features = {"a": 1.0, "c": 3.0}  # Missing 'b'
        feature_order = ["a", "b", "c"]

        vector = assemble_feature_vector(features, feature_order)

        assert vector[0, 1] == 0.0  # b defaults to 0

    def test_output_shape(self, sample_feature_order):
        """Output should be 2D array of shape (1, n_features)."""
        features = {name: float(i) for i, name in enumerate(sample_feature_order)}
        vector = assemble_feature_vector(features, sample_feature_order)

        assert vector.shape == (1, len(sample_feature_order))

    def test_output_dtype(self, sample_feature_order):
        """Output should be float32."""
        features = {name: float(i) for i, name in enumerate(sample_feature_order)}
        vector = assemble_feature_vector(features, sample_feature_order)

        assert vector.dtype == np.float32


class TestScoreTransaction:
    """Tests for the full scoring pipeline."""

    def test_successful_prediction(
        self, sample_transaction, mock_model, mock_feature_store, sample_feature_order
    ):
        """Should return complete prediction result."""
        result = score_transaction(
            tx=sample_transaction,
            model=mock_model,
            feature_order=sample_feature_order,
            threshold=0.5,
            model_version="test_v1",
            feature_store=mock_feature_store
        )

        assert "transaction_id" in result
        assert "customer_id" in result
        assert "terminal_id" in result
        assert "amount" in result
        assert "fraud_probability" in result
        assert "is_fraud" in result
        assert "risk_level" in result
        assert "threshold" in result
        assert "model_version" in result
        assert "timestamp" in result
        assert "features_used" in result

    def test_fraud_probability_range(
        self, sample_transaction, mock_model, mock_feature_store, sample_feature_order
    ):
        """Fraud probability should be between 0 and 1."""
        result = score_transaction(
            tx=sample_transaction,
            model=mock_model,
            feature_order=sample_feature_order,
            threshold=0.5,
            model_version="test_v1",
            feature_store=mock_feature_store
        )

        assert 0.0 <= result["fraud_probability"] <= 1.0

    def test_is_fraud_above_threshold(
        self, sample_transaction, mock_fraud_model, mock_feature_store, sample_feature_order
    ):
        """is_fraud should be True when probability >= threshold."""
        result = score_transaction(
            tx=sample_transaction,
            model=mock_fraud_model,
            feature_order=sample_feature_order,
            threshold=0.5,
            model_version="test_v1",
            feature_store=mock_feature_store
        )

        assert result["is_fraud"] is True

    def test_is_fraud_below_threshold(
        self, sample_transaction, mock_model, mock_feature_store, sample_feature_order
    ):
        """is_fraud should be False when probability < threshold."""
        result = score_transaction(
            tx=sample_transaction,
            model=mock_model,
            feature_order=sample_feature_order,
            threshold=0.5,
            model_version="test_v1",
            feature_store=mock_feature_store
        )

        assert result["is_fraud"] is False

    @pytest.mark.parametrize("prob,expected_level", [
        (0.95, "CRITICAL"),
        (0.85, "HIGH"),
        (0.65, "MEDIUM"),
        (0.35, "LOW"),
    ])
    def test_risk_level_categorization(
        self, sample_transaction, mock_feature_store, sample_feature_order, prob, expected_level
    ):
        """Risk levels should be categorized correctly."""
        model = Mock()
        model.predict_proba.return_value = np.array([[1 - prob, prob]])

        result = score_transaction(
            tx=sample_transaction,
            model=model,
            feature_order=sample_feature_order,
            threshold=0.5,
            model_version="test_v1",
            feature_store=mock_feature_store
        )

        assert result["risk_level"] == expected_level

    def test_invalid_transaction_raises(
        self, mock_model, mock_feature_store, sample_feature_order
    ):
        """Invalid transaction should raise ValueError."""
        invalid_tx = {
            "customer_id": 100,
            # Missing fields
        }

        with pytest.raises(ValueError):
            score_transaction(
                tx=invalid_tx,
                model=mock_model,
                feature_order=sample_feature_order,
                threshold=0.5,
                model_version="test_v1",
                feature_store=mock_feature_store
            )

    def test_uses_provided_transaction_id(
        self, sample_transaction, mock_model, mock_feature_store, sample_feature_order
    ):
        """Should use transaction_id from input if provided."""
        result = score_transaction(
            tx=sample_transaction,
            model=mock_model,
            feature_order=sample_feature_order,
            threshold=0.5,
            model_version="test_v1",
            feature_store=mock_feature_store
        )

        assert result["transaction_id"] == "test_tx_001"

    def test_generates_transaction_id_if_missing(
        self, mock_model, mock_feature_store, sample_feature_order
    ):
        """Should generate transaction_id if not provided."""
        tx = {
            "customer_id": 100,
            "terminal_id": 500,
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"
        }

        result = score_transaction(
            tx=tx,
            model=mock_model,
            feature_order=sample_feature_order,
            threshold=0.5,
            model_version="test_v1",
            feature_store=mock_feature_store
        )

        assert result["transaction_id"].startswith("tx_")

    def test_features_used_in_result(
        self, sample_transaction, mock_model, mock_feature_store, sample_feature_order
    ):
        """Result should include key features used for prediction."""
        result = score_transaction(
            tx=sample_transaction,
            model=mock_model,
            feature_order=sample_feature_order,
            threshold=0.5,
            model_version="test_v1",
            feature_store=mock_feature_store
        )

        features_used = result["features_used"]
        assert "customer_avg_amount_7d" in features_used
        assert "terminal_fraud_rate_7d" in features_used
        assert "amount_deviation" in features_used
        assert "risk_score" in features_used
