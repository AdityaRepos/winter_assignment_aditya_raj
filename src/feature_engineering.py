import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = None
    
    def create_merchant_features(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """Create merchant-level features from transaction data"""
        # Create a copy of the DataFrame to avoid warnings
        transactions_df = transactions_df.copy()
        
        # Convert timestamp to datetime if it's not already
        transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
        
        # Pre-calculate hour for all transactions
        transactions_df['hour'] = transactions_df['timestamp'].dt.hour
        
        merchant_features = []
        
        for merchant_id in transactions_df['merchant_id'].unique():
            # Create a copy of merchant transactions
            merchant_txns = transactions_df[transactions_df['merchant_id'].astype(str) == str(merchant_id)].copy()
            
            # Calculate features
            hour_dist = self._calculate_hourly_distribution(merchant_txns)
            daily_stats = self._calculate_daily_stats(merchant_txns)
            amount_stats = self._calculate_amount_stats(merchant_txns)
            customer_stats = self._calculate_customer_stats(merchant_txns)
            
            features = {
                'merchant_id': merchant_id,
                **hour_dist,
                **daily_stats,
                **amount_stats,
                **customer_stats
            }
            merchant_features.append(features)
        
        return pd.DataFrame(merchant_features)
    
    def _calculate_hourly_distribution(self, txns: pd.DataFrame) -> Dict:
        """Calculate hourly transaction distribution features"""
        # No need to set hour here as it's already set in create_merchant_features
        business_hours = (txns['hour'] >= 8) & (txns['hour'] <= 20)
        night_hours = txns['hour'].isin([23, 0, 1, 2, 3, 4])
        
        hourly_counts = txns.groupby('hour').size()
        total_txns = len(txns)
        
        return {
            'business_hour_ratio': sum(business_hours) / total_txns if total_txns > 0 else 0,
            'night_hour_ratio': sum(night_hours) / total_txns if total_txns > 0 else 0,
            'peak_hour_txns': hourly_counts.max() / total_txns if total_txns > 0 else 0
        }
    
    def _calculate_daily_stats(self, txns: pd.DataFrame) -> Dict:
        """Calculate daily transaction statistics"""
        daily_counts = txns.groupby(txns['timestamp'].dt.date).size()
        
        if len(daily_counts) == 0:
            return {
                'avg_daily_txns': 0,
                'max_daily_txns': 0,
                'daily_txn_std': 0,
                'volume_stability': 0
            }
        
        mean_daily = daily_counts.mean()
        return {
            'avg_daily_txns': mean_daily,
            'max_daily_txns': daily_counts.max(),
            'daily_txn_std': daily_counts.std() if len(daily_counts) > 1 else 0,
            'volume_stability': daily_counts.std() / mean_daily if mean_daily > 0 else 0
        }
    
    def _calculate_amount_stats(self, txns: pd.DataFrame) -> Dict:
        """Calculate transaction amount statistics"""
        if len(txns) == 0:
            return {
                'avg_amount': 0,
                'amount_std': 0,
                'amount_diversity': 0,
                'round_amount_ratio': 0
            }
        
        amounts = txns['amount']
        round_amounts = amounts[amounts.astype(str).str.endswith(('000', '999'))]
        
        return {
            'avg_amount': amounts.mean(),
            'amount_std': amounts.std() if len(amounts) > 1 else 0,
            'amount_diversity': amounts.nunique() / len(amounts),
            'round_amount_ratio': len(round_amounts) / len(amounts)
        }
    
    def _calculate_customer_stats(self, txns: pd.DataFrame) -> Dict:
        """Calculate customer-related statistics"""
        if len(txns) == 0:
            return {
                'customer_diversity': 0,
                'top_customer_ratio': 0,
                'avg_txns_per_customer': 0
            }
        
        customer_counts = txns['customer_id'].value_counts()
        total_customers = customer_counts.nunique()
        
        return {
            'customer_diversity': total_customers / len(txns),
            'top_customer_ratio': customer_counts.nlargest(5).sum() / len(txns) if len(customer_counts) > 0 else 0,
            'avg_txns_per_customer': len(txns) / total_customers if total_customers > 0 else 0
        }
    
    def prepare_training_data(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare and normalize features for model training"""
        # Select feature columns (exclude merchant_id)
        self.feature_columns = [col for col in features_df.columns if col != 'merchant_id']
        
        # Fit scaler on training data
        X = self.scaler.fit_transform(features_df[self.feature_columns])
        return X
    
    def prepare_test_data(self, features_df: pd.DataFrame) -> np.ndarray:
        """Prepare and normalize features for testing"""
        if self.feature_columns is None:
            raise ValueError("Scaler not fitted. Run prepare_training_data first.")
        
        # Transform test data using fitted scaler
        X = self.scaler.transform(features_df[self.feature_columns])
        return X 