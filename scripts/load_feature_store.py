"""Load the simulated transactions into the Supabase feature store.

Writes three tables: customer_features and terminal_features as of the last day
in the data, and the last 60 days of transaction_history (enough for a 30-day
window plus the 7-day label delay). Deletes existing feature rows first.

Usage: python scripts/load_feature_store.py [path/to/transactions_full.csv]
"""
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.features import LABEL_DELAY_DAYS, WINDOWS  # noqa: E402

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

HISTORY_DAYS = 60
BATCH_SIZE = 500


def _between(df: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return df[(df['TX_DATETIME'] >= start) & (df['TX_DATETIME'] < end)]


def _counts_as_int(frame: pd.DataFrame) -> pd.DataFrame:
    counts = [c for c in frame.columns if c.startswith(('tx_count_', 'fraud_count_'))]
    return frame.astype({c: int for c in counts})


def customer_snapshot(df: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
    """Each customer's spending over the 1, 7 and 30 days before `as_of`."""
    out = pd.DataFrame(index=pd.Index(df['CUSTOMER_ID'].unique(), name='customer_id'))
    for w in WINDOWS:
        amounts = _between(df, as_of - timedelta(days=w), as_of).groupby('CUSTOMER_ID')['TX_AMOUNT']
        out[f'tx_count_{w}d'] = amounts.count()
        out[f'avg_amount_{w}d'] = amounts.mean()
        out[f'std_amount_{w}d'] = amounts.std()
    last = df.sort_values('TX_DATETIME').groupby('CUSTOMER_ID').tail(1).set_index('CUSTOMER_ID')
    out['last_tx_timestamp'] = last['TX_DATETIME'].map(pd.Timestamp.isoformat)
    out['last_tx_amount'] = last['TX_AMOUNT']
    out = out.fillna({c: 0 for c in out.columns if c.startswith(('tx_', 'avg_', 'std_'))})
    return _counts_as_int(out).reset_index()


def terminal_snapshot(df: pd.DataFrame, as_of: pd.Timestamp, delay_days: int = LABEL_DELAY_DAYS) -> pd.DataFrame:
    """Each terminal's volume over the last 1/7/30 days, and fraud rate from labels at least `delay_days` old."""
    out = pd.DataFrame(index=pd.Index(df['TERMINAL_ID'].unique(), name='terminal_id'))
    labelled_until = as_of - timedelta(days=delay_days)
    for w in WINDOWS:
        amounts = _between(df, as_of - timedelta(days=w), as_of).groupby('TERMINAL_ID')['TX_AMOUNT']
        labels = _between(df, labelled_until - timedelta(days=w), labelled_until).groupby('TERMINAL_ID')['TX_FRAUD']
        out[f'tx_count_{w}d'] = amounts.count()
        out[f'avg_amount_{w}d'] = amounts.mean()
        out[f'fraud_count_{w}d'] = labels.sum()
        out[f'fraud_rate_{w}d'] = labels.mean()
    out = _counts_as_int(out.fillna(0))
    out['high_risk'] = out['fraud_rate_7d'] > 0.05
    return out.reset_index()


def _records(frame: pd.DataFrame) -> list:
    """Rows as plain Python values (numpy ints and floats don't serialize to JSON)."""
    frame = frame.assign(last_updated=datetime.now().isoformat())
    return [{k: (v.item() if isinstance(v, np.generic) else v) for k, v in row.items()}
            for row in frame.to_dict('records')]


def _write(client, table: str, rows: list, upsert: bool = True) -> None:
    for i in range(0, len(rows), BATCH_SIZE):
        batch = client.table(table)
        (batch.upsert if upsert else batch.insert)(rows[i:i + BATCH_SIZE]).execute()
    logger.info('%s: wrote %s rows', table, f'{len(rows):,}')


def main() -> None:
    csv = sys.argv[1] if len(sys.argv) > 1 else ROOT / 'data' / 'raw' / 'transactions_full.csv'
    df = pd.read_csv(csv, parse_dates=['TX_DATETIME'])
    as_of = df['TX_DATETIME'].max()
    logger.info('%s transactions, features as of %s', f'{len(df):,}', as_of)

    answer = input('This deletes the existing customer and terminal features in Supabase. Continue? (yes/no): ')
    if answer.strip().lower() != 'yes':
        print('Cancelled.')
        return

    from src.feature_store.store import FeatureStore
    client = FeatureStore().client
    client.table('customer_features').delete().neq('customer_id', -1).execute()
    client.table('terminal_features').delete().neq('terminal_id', -1).execute()

    _write(client, 'customer_features', _records(customer_snapshot(df, as_of)))
    _write(client, 'terminal_features', _records(terminal_snapshot(df, as_of)))

    history = df[df['TX_DATETIME'] >= as_of - timedelta(days=HISTORY_DAYS)]
    _write(client, 'transaction_history', [{
        'transaction_id': f'tx_{row.TRANSACTION_ID}',
        'customer_id': int(row.CUSTOMER_ID),
        'terminal_id': int(row.TERMINAL_ID),
        'amount': float(row.TX_AMOUNT),
        'timestamp': row.TX_DATETIME.isoformat(),
        'is_fraud': bool(row.TX_FRAUD),
        'fraud_probability': 0.0,
    } for row in history.itertuples()], upsert=False)


if __name__ == '__main__':
    main()
