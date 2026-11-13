"""Guards against the label leak: features may only use information known at prediction time."""
import pandas as pd
import pytest

from src.features import add_customer_features, add_terminal_features


def _transactions(rows):
    return pd.DataFrame(rows, columns=['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD']) \
             .assign(TX_DATETIME=lambda d: pd.to_datetime(d['TX_DATETIME']))


def _row(df, when):
    return df[df['TX_DATETIME'] == pd.Timestamp(when)].iloc[0]


def test_recent_fraud_is_not_yet_known():
    # Fraud 3 days earlier at the same terminal is still inside the 7-day label delay.
    df = add_terminal_features(_transactions([
        ('2018-04-10 10:00', 1, 9, 50.0, 1),
        ('2018-04-13 10:00', 2, 9, 50.0, 0),
    ]))
    assert _row(df, '2018-04-13 10:00')['TERMINAL_FRAUD_RATE_7D'] == 0.0


def test_fraud_older_than_delay_is_counted():
    # Fraud 9 days earlier is confirmed, and falls in the 7-day window that ends at the delay cutoff.
    df = add_terminal_features(_transactions([
        ('2018-04-01 10:00', 1, 9, 50.0, 1),
        ('2018-04-02 10:00', 3, 9, 50.0, 0),
        ('2018-04-10 10:00', 2, 9, 50.0, 0),
    ]))
    assert _row(df, '2018-04-10 10:00')['TERMINAL_FRAUD_RATE_7D'] == pytest.approx(0.5)


def test_windows_are_calendar_days_not_rows():
    # Two purchases 10 days apart: the 7-day window must be empty, whatever the row count.
    df = add_customer_features(_transactions([
        ('2018-04-01 10:00', 1, 9, 20.0, 0),
        ('2018-04-11 10:00', 1, 9, 80.0, 0),
    ]))
    later = _row(df, '2018-04-11 10:00')
    assert later['CUSTOMER_TX_COUNT_7D'] == 0
    assert later['CUSTOMER_TX_COUNT_30D'] == 1


def test_current_transaction_excluded_from_its_own_features():
    df = add_customer_features(_transactions([('2018-04-01 10:00', 1, 9, 20.0, 0)]))
    assert df.iloc[0]['CUSTOMER_TX_COUNT_1D'] == 0


class _FakeStore:
    """Stands in for the Supabase feature store with fixed history."""

    def get_customer_features(self, customer_id):
        return {f'CUSTOMER_{stat}_{w}D': value
                for w in (1, 7, 30)
                for stat, value in (('TX_COUNT', 3), ('AVG_AMOUNT', 40.0), ('STD_AMOUNT', 10.0))}

    def get_terminal_features(self, terminal_id):
        row = {f'TERMINAL_{stat}_{w}D': value
               for w in (1, 7, 30)
               for stat, value in (('TX_COUNT', 5), ('FRAUD_RATE', 0.0), ('AVG_AMOUNT', 60.0))}
        row['TERMINAL_HIGH_RISK'] = 0
        return row


def _served_features(amount):
    from src.model_utils import enrich_with_feature_store, extract_realtime_features
    tx = {'customer_id': 1, 'terminal_id': 9, 'amount': amount, 'timestamp': '2018-04-11T10:00:00'}
    return enrich_with_feature_store(tx, extract_realtime_features(tx), _FakeStore())


def test_serving_produces_every_model_feature():
    # assemble_feature_vector fills a missing feature with 0, so a gap here would be silent.
    from src.features import FEATURE_COLUMNS
    missing = [name for name in FEATURE_COLUMNS if name not in _served_features(100.0)]
    assert missing == []


def test_serving_ratio_matches_training_rule():
    served = _served_features(100.0)
    assert served['AMOUNT_TO_CUSTOMER_AVG_7D'] == pytest.approx(100.0 / 40.0)

    trained = add_customer_features(_transactions([
        ('2018-04-10 10:00', 1, 9, 40.0, 0),
        ('2018-04-11 10:00', 1, 9, 100.0, 0),
    ]))
    assert _row(trained, '2018-04-11 10:00')['AMOUNT_TO_CUSTOMER_AVG_7D'] == pytest.approx(100.0 / 40.0)
    # No history: both sides treat the purchase as normal spend.
    assert _row(trained, '2018-04-10 10:00')['AMOUNT_TO_CUSTOMER_AVG_7D'] == 1.0
