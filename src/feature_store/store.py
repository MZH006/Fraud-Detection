"""
Feature store implementation using Supabase for fraud detection.
Provides low-latency access to precomputed customer and terminal features.
"""

import os
from datetime import datetime, timedelta

LABEL_DELAY_DAYS = 7  # keep in sync with src/features.py
from typing import Dict, Optional, List, Any, cast
import pandas as pd
from supabase import create_client, Client
from dotenv import load_dotenv

from config.logging_config import get_logger

load_dotenv()

logger = get_logger(__name__)

class FeatureStore:
    """
    Feature store using Supabase for storing and retrieving precomputed features.

    Provides low-latency access to customer and terminal aggregated features
    (transaction counts, average amounts, fraud rates, etc.) that are used
    for real-time fraud detection.

    Attributes:
        client: Supabase client instance
        url: Supabase project URL
        key: Supabase API key

    Example:
        >>> store = FeatureStore()
        >>> features = store.get_customer_features(customer_id=123)
        >>> features["CUSTOMER_AVG_AMOUNT_7D"]
        75.5
    """

    def __init__(self):
        """
        Initialize feature store with Supabase connection.

        Raises:
            ValueError: If SUPABASE_URL or SUPABASE_KEY not set in environment
        """
        self.url = os.getenv("SUPABASE_URL")
        self.key = os.getenv("SUPABASE_KEY")

        if not self.url or not self.key:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in .env")

        self.client: Client = create_client(self.url, self.key)
        logger.info("Feature Store connected to Supabase")
    

    def get_customer_features(self, customer_id: int) -> Dict[str, float]:
        """
        Retrieve customer features from feature store.
        Returns default values if customer not found.
        """
        try:
            response = self.client.table('customer_features').select('*').eq('customer_id', customer_id).execute()
            
            if response.data and len(response.data) > 0:
                feature_row = cast(Dict[str, Any], response.data[0])
                
                return {
                    'CUSTOMER_TX_COUNT_1D': int(feature_row.get('tx_count_1d', 0) or 0),
                    'CUSTOMER_TX_COUNT_7D': int(feature_row.get('tx_count_7d', 0) or 0),
                    'CUSTOMER_TX_COUNT_30D': int(feature_row.get('tx_count_30d', 0) or 0),
                    'CUSTOMER_AVG_AMOUNT_1D': float(feature_row.get('avg_amount_1d', 0) or 0),
                    'CUSTOMER_AVG_AMOUNT_7D': float(feature_row.get('avg_amount_7d', 0) or 0),
                    'CUSTOMER_AVG_AMOUNT_30D': float(feature_row.get('avg_amount_30d', 0) or 0),
                    'CUSTOMER_STD_AMOUNT_1D': float(feature_row.get('std_amount_1d', 0) or 0),
                    'CUSTOMER_STD_AMOUNT_7D': float(feature_row.get('std_amount_7d', 0) or 0),
                    'CUSTOMER_STD_AMOUNT_30D': float(feature_row.get('std_amount_30d', 0) or 0),
                }
            else:
                return self._get_default_customer_features()
                
        except Exception as e:
            logger.error(f"Error fetching customer features: {e}")
            return self._get_default_customer_features()
        

    def get_terminal_features(self, terminal_id: int) -> Dict[str, float]:
        """
        Retrieve terminal features from feature store.
        Returns default values if terminal not found.
        """
        try:
            response = self.client.table('terminal_features').select('*').eq('terminal_id', terminal_id).execute()
            
            if response.data and len(response.data) > 0:
                feature_row = cast(Dict[str, Any], response.data[0])
                
                return {
                    'TERMINAL_TX_COUNT_1D': int(feature_row.get('tx_count_1d', 0) or 0),
                    'TERMINAL_TX_COUNT_7D': int(feature_row.get('tx_count_7d', 0) or 0),
                    'TERMINAL_TX_COUNT_30D': int(feature_row.get('tx_count_30d', 0) or 0),
                    'TERMINAL_FRAUD_RATE_1D': float(feature_row.get('fraud_rate_1d', 0) or 0),
                    'TERMINAL_FRAUD_RATE_7D': float(feature_row.get('fraud_rate_7d', 0) or 0),
                    'TERMINAL_FRAUD_RATE_30D': float(feature_row.get('fraud_rate_30d', 0) or 0),
                    'TERMINAL_AVG_AMOUNT_1D': float(feature_row.get('avg_amount_1d', 0) or 0),
                    'TERMINAL_AVG_AMOUNT_7D': float(feature_row.get('avg_amount_7d', 0) or 0),
                    'TERMINAL_AVG_AMOUNT_30D': float(feature_row.get('avg_amount_30d', 0) or 0),
                    'TERMINAL_HIGH_RISK': 1 if feature_row.get('high_risk', False) else 0,
                }
            else:
                return self._get_default_terminal_features()
                
        except Exception as e:
            logger.error(f"Error fetching terminal features: {e}")
            return self._get_default_terminal_features()
        

    def _get_default_customer_features(self) -> Dict[str, float]:
        """Default features for new customers"""
        return {
            'CUSTOMER_TX_COUNT_1D': 0,
            'CUSTOMER_TX_COUNT_7D': 0,
            'CUSTOMER_TX_COUNT_30D': 0,
            'CUSTOMER_AVG_AMOUNT_1D': 50.0,
            'CUSTOMER_AVG_AMOUNT_7D': 50.0,
            'CUSTOMER_AVG_AMOUNT_30D': 50.0,
            'CUSTOMER_STD_AMOUNT_1D': 30.0,
            'CUSTOMER_STD_AMOUNT_7D': 30.0,
            'CUSTOMER_STD_AMOUNT_30D': 30.0,
        }


    def _get_default_terminal_features(self) -> Dict[str, float]:
        """Default features for new terminals"""
        return {
            'TERMINAL_TX_COUNT_1D': 0,
            'TERMINAL_TX_COUNT_7D': 0,
            'TERMINAL_TX_COUNT_30D': 0,
            'TERMINAL_FRAUD_RATE_1D': 0.0,
            'TERMINAL_FRAUD_RATE_7D': 0.0,
            'TERMINAL_FRAUD_RATE_30D': 0.0,
            'TERMINAL_AVG_AMOUNT_1D': 50.0,
            'TERMINAL_AVG_AMOUNT_7D': 50.0,
            'TERMINAL_AVG_AMOUNT_30D': 50.0,
            'TERMINAL_HIGH_RISK': 0,
        }
    
    def log_transaction(self, transaction: Dict) -> bool:
        """
        Log transaction to history table.
        """
        try:
            data = {
                'transaction_id': transaction.get('transaction_id', f"tx_{datetime.now().timestamp()}"),
                'customer_id': transaction['customer_id'],
                'terminal_id': transaction['terminal_id'],
                'amount': transaction['amount'],
                'timestamp': transaction['timestamp'],
                'is_fraud': transaction.get('is_fraud', False),
                'fraud_probability': transaction.get('fraud_probability', 0.0)
            }
            
            self.client.table('transaction_history').insert(data).execute()
            return True
            
        except Exception as e:
            logger.error(f"Error logging transaction: {e}")
            return False
    
    def update_customer_features(self, customer_id: int):

        cutoff_date = datetime.now() - timedelta(days=30)

        try:
            response = self.client.table('transaction_history')\
                .select('*')\
                .eq('customer_id', customer_id)\
                .gte('timestamp', cutoff_date.isoformat())\
                .order('timestamp', desc=True)\
                .execute()
            if not response.data: 
                return

            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['amount'] = df['amount'].astype(float)

            now = datetime.now()
            features = {}
            features['customer_id'] = customer_id
            features['last_updated'] = now.isoformat()

            for window in [1, 7, 30]:
                window_start = now - timedelta(days=window)
                window_transactions = df[df['timestamp'] >= window_start]

                features[f'tx_count_{window}d'] = len(window_transactions)
                features[f'avg_amount_{window}d'] = float(window_transactions['amount'].mean()) if len(window_transactions) > 0 else 0.0
                features[f'std_amount_{window}d'] = float(window_transactions['amount'].std()) if len(window_transactions) > 0 else 0.0

            if len(df) > 0:
                features['last_tx_timestamp'] = df.iloc[0]['timestamp'].isoformat()
                features['last_tx_amount'] = float(df.iloc[0]['amount'])
            
            self.client.table('customer_features').upsert(features).execute()
            logger.info(f"Updated customer {customer_id} features")

        except Exception as e:
            logger.error(f"Error updating customer features: {e}")

    def update_terminal_features(self, terminal_id: int):
        """
        Recompute terminal features from recent transaction history.
        Called by background worker.
        """
        try:
            # Fetch 30 days plus the label delay so the delayed 30-day fraud rate is complete.
            cutoff = datetime.now() - timedelta(days=30 + LABEL_DELAY_DAYS)
            
            response = self.client.table('transaction_history')\
                .select('*')\
                .eq('terminal_id', terminal_id)\
                .gte('timestamp', cutoff.isoformat())\
                .order('timestamp', desc=True)\
                .execute()
            
            if not response.data:
                return
            
            df = pd.DataFrame(response.data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['amount'] = df['amount'].astype(float)
            df['is_fraud'] = df['is_fraud'].astype(bool)
            
            now = datetime.now()
            
            features = {
                'terminal_id': terminal_id,
                'last_updated': now.isoformat(),
            }
            
            # Fraud labels are only confirmed after LABEL_DELAY_DAYS, so fraud rates use
            # [now - (delay + days), now - delay), matching src/features.py at training time.
            label_cutoff = now - timedelta(days=LABEL_DELAY_DAYS)
            for days in [1, 7, 30]:
                window_start = now - timedelta(days=days)
                window_df = df[df['timestamp'] >= window_start]
                labelled_df = df[(df['timestamp'] >= label_cutoff - timedelta(days=days))
                                 & (df['timestamp'] < label_cutoff)]

                tx_count = len(window_df)
                labelled_count = len(labelled_df)
                fraud_count = labelled_df['is_fraud'].sum()

                features[f'tx_count_{days}d'] = tx_count
                features[f'fraud_count_{days}d'] = int(fraud_count)
                features[f'fraud_rate_{days}d'] = float(fraud_count / labelled_count) if labelled_count > 0 else 0.0
                features[f'avg_amount_{days}d'] = float(window_df['amount'].mean()) if tx_count > 0 else 0
            
            # High risk flag (fraud rate > 5% in last 7 days)
            features['high_risk'] = features['fraud_rate_7d'] > 0.05
            
            self.client.table('terminal_features').upsert(features).execute()
            logger.info(f"Updated terminal {terminal_id} features")
            
        except Exception as e:
            logger.error(f"Error updating terminal features: {e}")

