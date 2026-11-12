#IMPORTANT RUN ONCE ONLYYYYY

########################################################################################
# due to my carelessness it seems i had a bug in the code and i have to rerun everything
#######################################################################################

import sys
import os
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from tqdm import tqdm
import logging

from feature_store.store import FeatureStore

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureStoreInitializer:
    """
    Intialize historical transaction data mainly for my testing purpose
    """

    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df
        self.feature_store = FeatureStore()
        logger.info(f"initializer loaded with {len(df):,} transactions")

    def populate_customer_features(self, batch_size: int = 100):
        """
        Compute and store features for the customers
        """

        logger.info("="*60)
        logger.info("POPULATING CUSTOMER FEATURES")
        logger.info("="*60)

        unique_customers = self.df['CUSTOMER_ID'].unique()
        logger.info(f"Found {len(unique_customers):,} unique customers")

        customer_features_batch = []

        for customer_id in tqdm(unique_customers, desc='Processing customers'):
            try:
                customer_df = self.df[self.df['CUSTOMER_ID'] == customer_id].copy()
                customer_df = customer_df.sort_values('TX_DATETIME')

                last_date = customer_df['TX_DATETIME'].max()
                cutoff = last_date - timedelta(days=30)
                recent_df = customer_df[customer_df['TX_DATETIME'] >= cutoff]        

                if len(recent_df) == 0:
                    continue

                features = {
                    'customer_id': int(customer_id),
                    'last_updated': datetime.now().isoformat(),
                }        

                for window in [1, 7, 30]:
                    window_start = last_date - timedelta(days=window)
                    window_df = recent_df[recent_df['TX_DATETIME'] >= window_start]
                    
                    # Compute std, handle NaN
                    std_val = float(window_df['TX_AMOUNT'].std()) if len(window_df) > 1 else 0.0
                    
                    features[f'tx_count_{window}d'] = len(window_df)
                    features[f'avg_amount_{window}d'] = float(window_df['TX_AMOUNT'].mean()) if len(window_df) > 0 else 0.0
                    features[f'std_amount_{window}d'] = 0.0 if pd.isna(std_val) else std_val  # ✅ FIX
                
                features['last_tx_timestamp'] = recent_df.iloc[-1]['TX_DATETIME'].isoformat()
                features['last_tx_amount'] = float(recent_df.iloc[-1]['TX_AMOUNT'])

                customer_features_batch.append(features)

                if len(customer_features_batch) >= batch_size:
                    self._insert_customer_batch(customer_features_batch)
                    customer_features_batch = []

            except Exception as e:
                logger.error(f"Error processing customer {customer_id}: {e}")

        if customer_features_batch:
            self._insert_customer_batch(customer_features_batch)
        
        logger.info(f"✓ Populated {len(unique_customers):,} customer records")

    def _insert_customer_batch(self, batch: list):
        """Insert batch of customer features"""
        try:
            self.feature_store.client.table('customer_features').upsert(batch).execute()
        except Exception as e:
            logger.error(f"Error inserting customer batch: {e}")

    def populate_terminal_features(self, batch_size: int = 100):
        """
        Compute and store terminal features for all terminals.
        """
        logger.info("="*60)
        logger.info("POPULATING TERMINAL FEATURES")
        logger.info("="*60)
        
        unique_terminals = self.df['TERMINAL_ID'].unique()
        logger.info(f"{len(unique_terminals):,} unique terminals")
        
        terminal_features_batch = []
        
        for terminal_id in tqdm(unique_terminals, desc="Processing terminals"):
            try:
                # Get this terminal's transactions
                terminal_df = self.df[self.df['TERMINAL_ID'] == terminal_id].copy()
                terminal_df = terminal_df.sort_values('TX_DATETIME')
                
                # Use last 30 days of data
                last_date = terminal_df['TX_DATETIME'].max()
                cutoff = last_date - timedelta(days=30)
                recent_df = terminal_df[terminal_df['TX_DATETIME'] >= cutoff]
                
                if len(recent_df) == 0:
                    continue
                
                # Compute features
                features = {
                    'terminal_id': int(terminal_id),
                    'last_updated': datetime.now().isoformat(),
                }
                
                for window in [1, 7, 30]:
                    window_start = last_date - timedelta(days=window)
                    window_df = recent_df[recent_df['TX_DATETIME'] >= window_start]
                    
                    tx_count = len(window_df)
                    fraud_count = int(window_df['TX_FRAUD'].sum())
                    
                    features[f'tx_count_{window}d'] = tx_count
                    features[f'fraud_count_{window}d'] = fraud_count
                    features[f'fraud_rate_{window}d'] = float(fraud_count / tx_count) if tx_count > 0 else 0.0
                    features[f'avg_amount_{window}d'] = float(window_df['TX_AMOUNT'].mean()) if tx_count > 0 else 0.0
                
                # High risk flag
                features['high_risk'] = features['fraud_rate_7d'] > 0.05
                
                terminal_features_batch.append(features)
                
                # Batch insert
                if len(terminal_features_batch) >= batch_size:
                    self._insert_terminal_batch(terminal_features_batch)
                    terminal_features_batch = []
                    
            except Exception as e:
                logger.error(f"Error processing terminal {terminal_id}: {e}")
        
        # Insert remaining
        if terminal_features_batch:
            self._insert_terminal_batch(terminal_features_batch)
        
        logger.info(f"Populated {len(unique_terminals):,} terminal records")
    
    def _insert_terminal_batch(self, batch: list):
        """Insert batch of terminal features"""
        try:
            self.feature_store.client.table('terminal_features').upsert(batch).execute()
        except Exception as e:
            logger.error(f"Error inserting terminal batch: {e}")
    
    def populate_transaction_history(self, batch_size: int = 1000):
        """
        Load transaction history into the database.
        This enables the background worker to recompute features.
        
        Note: For large datasets (1M+ rows), consider only loading recent data.
        """
        logger.info("="*60)
        logger.info("POPULATING TRANSACTION HISTORY")
        logger.info("="*60)
        
        # Use only recent transactions to avoid overwhelming the DB
        # Last 60 days should be sufficient
        cutoff = self.df['TX_DATETIME'].max() - timedelta(days=60)
        recent_df = self.df[self.df['TX_DATETIME'] >= cutoff].copy()
        
        logger.info(f"Loading {len(recent_df):,} recent transactions (last 60 days)")
        
        # Prepare data for insertion
        transactions = []
        
        for _, row in tqdm(recent_df.iterrows(), total=len(recent_df), desc="Preparing transactions"):
            tx = {
                'transaction_id': f"tx_{row['TRANSACTION_ID']}",
                'customer_id': int(row['CUSTOMER_ID']),
                'terminal_id': int(row['TERMINAL_ID']),
                'amount': float(row['TX_AMOUNT']),
                'timestamp': row['TX_DATETIME'].isoformat(),
                'is_fraud': bool(row['TX_FRAUD']),
                'fraud_probability': 0.0  # Unknown for historical data
            }
            transactions.append(tx)
            
            # Batch insert
            if len(transactions) >= batch_size:
                self._insert_transaction_batch(transactions)
                transactions = []
        
        # Insert remaining
        if transactions:
            self._insert_transaction_batch(transactions)
        
        logger.info(f"✓ Populated {len(recent_df):,} transaction records")
    
    def _insert_transaction_batch(self, batch: list):
        """Insert batch of transactions"""
        try:
            self.feature_store.client.table('transaction_history').insert(batch).execute()
        except Exception as e:
            logger.error(f"Error inserting transaction batch: {e}")
    
    def run_full_initialization(self):
        """
        Run complete initialization process.
        """
        logger.info("\n" + "="*60)
        logger.info("STARTING FEATURE STORE INITIALIZATION")
        logger.info("="*60)
        
        start_time = datetime.now()
        
        #Populate customer features
        self.populate_customer_features(batch_size=100)
        
        #Populate terminal features
        self.populate_terminal_features(batch_size=100)
        
        #Populate transaction history
        self.populate_transaction_history(batch_size=1000)
        
        elapsed = datetime.now() - start_time
        
        logger.info("\n" + "="*60)
        logger.info("INITIALIZATION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total time: {elapsed}")

    
def clear_existing_data():
    """Clear existing data before re-initializing :("""
    logger.info("\n" + "="*60)
    logger.info("CLEARING EXISTING DATA")
    logger.info("="*60)
    
    from feature_store.store import FeatureStore
    fs = FeatureStore()
    
    try:
        # Delete customer features
        logger.info("Deleting customer features...")
        fs.client.table('customer_features').delete().neq('customer_id', -1).execute()
        
        # Delete terminal features
        logger.info("Deleting terminal features...")
        fs.client.table('terminal_features').delete().neq('terminal_id', -1).execute()
        
        # Transaction history already cleared manually in Supabase
        logger.info("Customer and terminal features cleared")
        logger.info("   (transaction_history cleared manually)")
        
        return True
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        return False

def main():
    """
    Main initialization script.
    Run this ONCE
    """
    # Build path relative to this script's location
    script_dir = Path(__file__).parent
    data_path = script_dir.parent.parent / 'data' / 'raw' / 'transactions_full.csv'
    df = pd.read_csv(data_path)
    
    # Ensure datetime column
    df['TX_DATETIME'] = pd.to_datetime(df['TX_DATETIME'])
    
    print(f"Loaded {len(df):,} transactions")
    print(f"Date range: {df['TX_DATETIME'].min()} to {df['TX_DATETIME'].max()}")
    print(f"Customers: {df['CUSTOMER_ID'].nunique():,}")
    print(f"Terminals: {df['TERMINAL_ID'].nunique():,}")
    print(f"Fraud rate: {df['TX_FRAUD'].mean():.2%}")
    
    # Clear existing data first
    print("\n⚠️  This will DELETE all existing data in Supabase and reinitialize.")
    response = input("Proceed with DELETION and reinitialization? (yes/no): ")
    if response.lower() != 'yes':
        print("Initialization cancelled.")
        return
    
    # Clear data
    if not clear_existing_data():
        print("Failed to clear data. Aborting.")
        return
    
    # Run initialization
    initializer = FeatureStoreInitializer(df)
    initializer.run_full_initialization()



if __name__ == '__main__':
    main()
