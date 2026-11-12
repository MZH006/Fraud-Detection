import pytest
from datetime import datetime
from src.model_utils import extract_realtime_features

class TestFeatureExtraction:
    """Test suite for feature extraction logic"""

    def test_extract_basic_features(self):
        """Test extraction of basic transaction features"""
        tx = {
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"
        }

        features = extract_realtime_features(tx)

        assert features["TX_AMOUNT"] == 100.0
        assert "TX_HOUR_SIN" in features
        assert "TX_HOUR_COS" in features
        assert features["TX_DAY_OF_WEEK"] in range(7)

    def test_high_amount_flag(self):
        """Test high amount threshold detection"""
        tx_high = {
            "amount": 250.0,
            "timestamp": "2024-01-15T14:30:00"
        }
        tx_low = {
            "amount": 50.0,
            "timestamp": "2024-01-15T14:30:00"
        }

        features_high = extract_realtime_features(tx_high)
        features_low = extract_realtime_features(tx_low)

        assert features_high["TX_AMOUNT_HIGH"] == 1
        assert features_low["TX_AMOUNT_HIGH"] == 0

    def test_night_transaction_detection(self):
        """Test night transaction flag (10pm-6am)"""
        tx_night = {
            "amount": 100.0,
            "timestamp": "2024-01-15T02:30:00"  # 2:30 AM
        }
        tx_day = {
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"  # 2:30 PM
        }

        features_night = extract_realtime_features(tx_night)
        features_day = extract_realtime_features(tx_day)

        assert features_night["TX_IS_NIGHT"] == 1
        assert features_day["TX_IS_NIGHT"] == 0

    def test_weekend_detection(self):
        """Test weekend flag detection"""
        tx_weekend = {
            "amount": 100.0,
            "timestamp": "2024-01-13T14:30:00"  # Saturday
        }
        tx_weekday = {
            "amount": 100.0,
            "timestamp": "2024-01-15T14:30:00"  # Monday
        }

        features_weekend = extract_realtime_features(tx_weekend)
        features_weekday = extract_realtime_features(tx_weekday)

        assert features_weekend["TX_IS_WEEKEND"] == 1
        assert features_weekday["TX_IS_WEEKEND"] == 0

    @pytest.mark.parametrize("invalid_timestamp", [
        "",
        "invalid",
        "2024-13-01T00:00:00",  # Invalid month
        "not-a-date"
    ])
    def test_invalid_timestamp(self, invalid_timestamp):
        """Test handling of invalid timestamps"""
        tx = {
            "amount": 100.0,
            "timestamp": invalid_timestamp
        }

        with pytest.raises(ValueError):
            extract_realtime_features(tx)

    def test_negative_amount(self):
        """Test that negative amounts are handled

        Note: extract_realtime_features is for feature extraction only.
        Validation of negative amounts is done in validate_schema().
        This test verifies that extraction works with any numeric amount.
        """
        tx = {
            "amount": -50.0,
            "timestamp": "2024-01-15T14:30:00"
        }

        # extract_realtime_features handles feature extraction, not validation
        # Negative amount validation is done by validate_schema()
        features = extract_realtime_features(tx)
        assert features["TX_AMOUNT"] == -50.0
        assert features["TX_AMOUNT_HIGH"] == 0  # -50 is not > 200