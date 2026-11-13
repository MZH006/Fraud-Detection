import numpy as np
import pandas as pd

LABEL_DELAY_DAYS = 7
WINDOWS = (1, 7, 30)

FEATURE_COLUMNS = [
    'TX_HOUR_SIN', 'TX_HOUR_COS', 'TX_DAY_OF_WEEK', 'TX_IS_WEEKEND', 'TX_IS_NIGHT',
    'TX_AMOUNT',
    'CUSTOMER_TX_COUNT_1D', 'CUSTOMER_TX_COUNT_7D', 'CUSTOMER_TX_COUNT_30D',
    'CUSTOMER_AVG_AMOUNT_1D', 'CUSTOMER_AVG_AMOUNT_7D', 'CUSTOMER_AVG_AMOUNT_30D',
    'CUSTOMER_STD_AMOUNT_1D', 'CUSTOMER_STD_AMOUNT_7D', 'CUSTOMER_STD_AMOUNT_30D',
    'CUSTOMER_AMOUNT_DEVIATION',
    'AMOUNT_TO_CUSTOMER_AVG_1D', 'AMOUNT_TO_CUSTOMER_AVG_7D', 'AMOUNT_TO_CUSTOMER_AVG_30D',
    'TERMINAL_TX_COUNT_1D', 'TERMINAL_TX_COUNT_7D', 'TERMINAL_TX_COUNT_30D',
    'TERMINAL_FRAUD_RATE_1D', 'TERMINAL_FRAUD_RATE_7D', 'TERMINAL_FRAUD_RATE_30D',
    'TERMINAL_AVG_AMOUNT_1D', 'TERMINAL_AVG_AMOUNT_7D', 'TERMINAL_AVG_AMOUNT_30D',
]


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    df['TX_HOUR'] = df['TX_DATETIME'].dt.hour
    df['TX_HOUR_SIN'] = np.sin(2 * np.pi * df['TX_HOUR'] / 24)
    df['TX_HOUR_COS'] = np.cos(2 * np.pi * df['TX_HOUR'] / 24)
    df['TX_DAY_OF_WEEK'] = df['TX_DATETIME'].dt.dayofweek
    df['TX_IS_WEEKEND'] = (df['TX_DAY_OF_WEEK'] >= 5).astype(int)
    df['TX_IS_NIGHT'] = ((df['TX_HOUR'] >= 22) | (df['TX_HOUR'] <= 6)).astype(int)
    return df


def _past_window(df: pd.DataFrame, key: str, col: str, window: str, agg: str) -> np.ndarray:
    """Per-key rolling stat over [t - window, t). `df` must be sorted by [key, TX_DATETIME]."""
    rolled = (df.set_index('TX_DATETIME')
                .groupby(key, sort=False)[col]
                .rolling(window, closed='left'))
    values = getattr(rolled, agg)().to_numpy()
    return np.nan_to_num(values) if agg in ('count', 'sum') else values


def add_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(['CUSTOMER_ID', 'TX_DATETIME'], kind='stable').reset_index(drop=True)
    for w in WINDOWS:
        df[f'CUSTOMER_TX_COUNT_{w}D'] = _past_window(df, 'CUSTOMER_ID', 'TX_AMOUNT', f'{w}D', 'count')
        df[f'CUSTOMER_AVG_AMOUNT_{w}D'] = _past_window(df, 'CUSTOMER_ID', 'TX_AMOUNT', f'{w}D', 'mean')
        df[f'CUSTOMER_STD_AMOUNT_{w}D'] = _past_window(df, 'CUSTOMER_ID', 'TX_AMOUNT', f'{w}D', 'std')
    # How many times the customer's usual spend this purchase is; 1.0 when there is no history.
    # Computed before the fill below, so "no history" means 1.0 here as it does at serving time.
    for w in WINDOWS:
        avg = df[f'CUSTOMER_AVG_AMOUNT_{w}D']
        df[f'AMOUNT_TO_CUSTOMER_AVG_{w}D'] = np.where(avg > 0, df['TX_AMOUNT'] / avg, 1.0)
    df['CUSTOMER_AVG_AMOUNT_7D'] = df['CUSTOMER_AVG_AMOUNT_7D'].fillna(df['TX_AMOUNT'].mean())
    df['CUSTOMER_STD_AMOUNT_7D'] = df['CUSTOMER_STD_AMOUNT_7D'].fillna(df['TX_AMOUNT'].std())
    df['CUSTOMER_AMOUNT_DEVIATION'] = (
        (df['TX_AMOUNT'] - df['CUSTOMER_AVG_AMOUNT_7D']) / (df['CUSTOMER_STD_AMOUNT_7D'] + 1e-5)
    ).fillna(0).clip(-10, 10)
    return df


def add_terminal_features(df: pd.DataFrame, delay_days: int = LABEL_DELAY_DAYS) -> pd.DataFrame:
    df = df.sort_values(['TERMINAL_ID', 'TX_DATETIME'], kind='stable').reset_index(drop=True)
    unlabelled_frauds = _past_window(df, 'TERMINAL_ID', 'TX_FRAUD', f'{delay_days}D', 'sum')
    unlabelled_count = _past_window(df, 'TERMINAL_ID', 'TX_FRAUD', f'{delay_days}D', 'count')
    for w in WINDOWS:
        df[f'TERMINAL_TX_COUNT_{w}D'] = _past_window(df, 'TERMINAL_ID', 'TX_AMOUNT', f'{w}D', 'count')
        df[f'TERMINAL_AVG_AMOUNT_{w}D'] = _past_window(df, 'TERMINAL_ID', 'TX_AMOUNT', f'{w}D', 'mean')
        # Labels known today cover [t - (delay + w), t - delay).
        frauds = _past_window(df, 'TERMINAL_ID', 'TX_FRAUD', f'{delay_days + w}D', 'sum') - unlabelled_frauds
        count = _past_window(df, 'TERMINAL_ID', 'TX_FRAUD', f'{delay_days + w}D', 'count') - unlabelled_count
        with np.errstate(invalid='ignore', divide='ignore'):
            df[f'TERMINAL_FRAUD_RATE_{w}D'] = np.where(count > 0, frauds / count, 0.0)
    df['TERMINAL_AVG_AMOUNT_7D'] = df['TERMINAL_AVG_AMOUNT_7D'].fillna(df['TX_AMOUNT'].mean())
    return df


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(raw)
    df = add_terminal_features(add_customer_features(df))
    return df.sort_values('TX_DATETIME', kind='stable').reset_index(drop=True)
