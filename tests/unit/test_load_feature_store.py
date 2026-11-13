"""The loader's snapshot must follow the same rules as training: shared as-of date, delayed labels."""
import pandas as pd
import pytest

from scripts.load_feature_store import customer_snapshot, terminal_snapshot

AS_OF = pd.Timestamp('2018-04-20 00:00')


def _transactions(rows):
    return pd.DataFrame(rows, columns=['TX_DATETIME', 'CUSTOMER_ID', 'TERMINAL_ID', 'TX_AMOUNT', 'TX_FRAUD']) \
             .assign(TX_DATETIME=lambda d: pd.to_datetime(d['TX_DATETIME']))


def _one(frame, key, value):
    return frame[frame[key] == value].iloc[0]


def test_recent_fraud_not_counted_in_terminal_rate():
    df = _transactions([('2018-04-18 10:00', 1, 9, 50.0, 1)])  # 2 days before as-of: label not known yet
    assert _one(terminal_snapshot(df, AS_OF), 'terminal_id', 9)['fraud_rate_7d'] == 0.0


def test_confirmed_fraud_counted_in_terminal_rate():
    df = _transactions([
        ('2018-04-09 10:00', 1, 9, 50.0, 1),   # 10 days old: inside [as_of - 14d, as_of - 7d)
        ('2018-04-10 10:00', 2, 9, 50.0, 0),
    ])
    assert _one(terminal_snapshot(df, AS_OF), 'terminal_id', 9)['fraud_rate_7d'] == pytest.approx(0.5)


def test_windows_share_one_as_of_date():
    # Customer 2's only purchase was 20 days ago, so their 7-day window is empty.
    df = _transactions([
        ('2018-04-19 10:00', 1, 9, 30.0, 0),
        ('2018-03-31 10:00', 2, 9, 80.0, 0),
    ])
    snap = customer_snapshot(df, AS_OF)
    assert _one(snap, 'customer_id', 2)['tx_count_7d'] == 0
    assert _one(snap, 'customer_id', 2)['tx_count_30d'] == 1
    assert _one(snap, 'customer_id', 1)['avg_amount_7d'] == pytest.approx(30.0)


def test_counts_are_integers():
    # The database columns are integers; 5.0 can be rejected.
    df = _transactions([('2018-04-19 10:00', 1, 9, 30.0, 0)])
    assert isinstance(_one(customer_snapshot(df, AS_OF), 'customer_id', 1)['tx_count_7d'].item(), int)
    assert isinstance(_one(terminal_snapshot(df, AS_OF), 'terminal_id', 9)['fraud_count_7d'].item(), int)
