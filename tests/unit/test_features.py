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
