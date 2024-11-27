import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
import logging
from tqdm import tqdm
from anomaly_detection import AutoencoderAnomalyDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FraudDetectionSystem:
    def __init__(self, autoencoder=None):
        self.autoencoder = autoencoder
        self.pattern_thresholds = {
            'late_night': {
                'ratio_threshold': 0.4,
                'amount_threshold': 5000
            },
            'velocity': {
                'spike_threshold': 4.0,
                'volume_threshold': 200
            },
            'split': {
                'time_window_minutes': 30,
                'min_splits': 5,
                'amount_threshold': 10000
            },
            'round_amount': {
                'ratio_threshold': 0.3,
                'amount_patterns': ['999', '000']
            },
            'customer_concentration': {
                'concentration_threshold': 0.7,
                'min_transactions': 50
            }
        }

    def calculate_risk_score(self, merchant_patterns: Dict) -> float:
        """Calculate risk score with adjusted weights"""
        weights = {
            'late_night': 0.25,
            'velocity': 0.25,
            'split': 0.2,
            'round_amount': 0.15,
            'customer_concentration': 0.15
        }
        
        scores = []
        for pattern, weight in weights.items():
            if pattern in merchant_patterns:
                pattern_score = self._calculate_pattern_score(
                    pattern, 
                    merchant_patterns[pattern]
                )
                scores.append(pattern_score * weight)
        
        return sum(scores)

    def _calculate_pattern_score(self, pattern_type: str, 
                               pattern_metrics: Dict) -> float:
        """Calculate individual pattern scores"""
        thresholds = self.pattern_thresholds[pattern_type]
        
        if pattern_type == 'late_night':
            return (pattern_metrics['late_night_ratio'] > thresholds['ratio_threshold'] and 
                   pattern_metrics['avg_amount'] > thresholds['amount_threshold'])
        
        elif pattern_type == 'velocity':
            return (pattern_metrics['spike_ratio'] > thresholds['spike_threshold'] and 
                   pattern_metrics['max_volume'] > thresholds['volume_threshold'])
        
        elif pattern_type == 'split':
            return (pattern_metrics['split_ratio'] > 0 and 
                   pattern_metrics['split_count'] >= thresholds['min_splits'])
        
        elif pattern_type == 'round_amount':
            return pattern_metrics['round_ratio'] > thresholds['ratio_threshold']
        
        elif pattern_type == 'customer_concentration':
            return (pattern_metrics['concentration_ratio'] > thresholds['concentration_threshold'] and 
                   pattern_metrics['transaction_count'] > thresholds['min_transactions'])
        
        return 0.0

    def _detect_late_night_pattern(self, transactions_df: pd.DataFrame, 
                                 merchant_id: str) -> Dict[str, float]:
        """Detect late night trading patterns"""
        merchant_txns = transactions_df[transactions_df['merchant_id'] == merchant_id].copy()
        
        merchant_txns.loc[:, 'timestamp'] = pd.to_datetime(merchant_txns['timestamp'])
        merchant_txns.loc[:, 'hour'] = merchant_txns['timestamp'].dt.hour
        
        late_night_mask = (merchant_txns['hour'] >= 23) | (merchant_txns['hour'] <= 4)
        late_night_txns = merchant_txns[late_night_mask]
        
        total_txns = len(merchant_txns)
        late_night_count = len(late_night_txns)
        
        score = {
            'late_night_ratio': late_night_count / total_txns if total_txns > 0 else 0,
            'late_night_amount_avg': late_night_txns['amount'].mean() if late_night_count > 0 else 0
        }
        return score

    def _detect_velocity_pattern(self, transactions_df: pd.DataFrame, 
                               merchant_id: str) -> Dict[str, float]:
        """Detect sudden spikes in transaction velocity"""
        merchant_txns = transactions_df[transactions_df['merchant_id'] == merchant_id].copy()
        merchant_txns.loc[:, 'timestamp'] = pd.to_datetime(merchant_txns['timestamp'])
        merchant_txns.loc[:, 'date'] = merchant_txns['timestamp'].dt.date
        
        daily_counts = merchant_txns.groupby('date').size()
        mean_daily_count = daily_counts.mean()
        
        score = {
            'max_daily_transactions': daily_counts.max(),
            'avg_daily_transactions': mean_daily_count,
            'velocity_ratio': daily_counts.max() / mean_daily_count if mean_daily_count > 0 else 0
        }
        return score

    def _detect_split_transactions(self, transactions_df: pd.DataFrame, 
                                 merchant_id: str) -> Dict[str, float]:
        """Detect split transaction patterns"""
        merchant_txns = transactions_df[transactions_df['merchant_id'] == merchant_id].copy()
        merchant_txns.loc[:, 'timestamp'] = pd.to_datetime(merchant_txns['timestamp'])
        
        merchant_txns = merchant_txns.sort_values('timestamp')
        
        merchant_txns['time_diff'] = merchant_txns.groupby('customer_id')['timestamp'].diff()
        split_candidates = merchant_txns[merchant_txns['time_diff'] <= timedelta(minutes=30)]
        
        total_txns = len(merchant_txns)
        split_count = len(split_candidates)
        
        score = {
            'split_transaction_ratio': split_count / total_txns if total_txns > 0 else 0,
            'avg_split_amount': split_candidates['amount'].mean() if split_count > 0 else 0
        }
        return score

    def _detect_round_amounts(self, transactions_df: pd.DataFrame, 
                            merchant_id: str) -> Dict[str, float]:
        """Detect round amount patterns"""
        merchant_txns = transactions_df[transactions_df['merchant_id'] == merchant_id].copy()
        amounts = merchant_txns['amount']
        
        round_amount_mask = amounts.astype(str).str.endswith(('999', '000'))
        round_amounts = amounts[round_amount_mask]
        
        total_txns = len(amounts)
        round_count = len(round_amounts)
        
        score = {
            'round_amount_ratio': round_count / total_txns if total_txns > 0 else 0,
            'round_amount_avg': round_amounts.mean() if round_count > 0 else 0
        }
        return score

    def _detect_customer_concentration(self, transactions_df: pd.DataFrame, 
                                      merchant_id: str) -> Dict[str, float]:
        """Detect customer concentration patterns"""
        merchant_txns = transactions_df[transactions_df['merchant_id'] == merchant_id].copy()
        merchant_txns.loc[:, 'timestamp'] = pd.to_datetime(merchant_txns['timestamp'])
        merchant_txns.loc[:, 'date'] = merchant_txns['timestamp'].dt.date
        
        daily_counts = merchant_txns.groupby('date').size()
        mean_daily_count = daily_counts.mean()
        
        concentration_ratio = (daily_counts.max() / mean_daily_count) / 5
        
        score = {
            'concentration_ratio': concentration_ratio
        }
        return score

    def calculate_fraud_score(self, transactions: List[Dict], 
                            features: np.ndarray) -> Dict[str, Dict]:
        """Calculate comprehensive fraud score with adjusted weights"""
        transactions_df = pd.DataFrame(transactions)
        merchant_scores = {}

        reconstruction_errors = self.autoencoder.calculate_reconstruction_error(features)
        
        unique_merchants = transactions_df['merchant_id'].unique()

        # Adjusted weights for different patterns
        pattern_weights = {
            'late_night': 0.25,
            'velocity': 0.25,
            'split_transactions': 0.2,
            'round_amounts': 0.15,
            'customer_concentration': 0.15
        }

        for merchant_id in tqdm(unique_merchants, desc="Processing merchants"):
            try:
                merchant_scores[merchant_id] = {
                    'autoencoder_score': float(np.mean(reconstruction_errors)),
                    'late_night': self._detect_late_night_pattern(transactions_df, merchant_id),
                    'velocity': self._detect_velocity_pattern(transactions_df, merchant_id),
                    'split_transactions': self._detect_split_transactions(transactions_df, merchant_id),
                    'round_amounts': self._detect_round_amounts(transactions_df, merchant_id),
                    'customer_concentration': self._detect_customer_concentration(transactions_df, merchant_id)
                }
                
                # Calculate weighted risk score
                risk_score = (
                    merchant_scores[merchant_id]['late_night']['late_night_ratio'] * pattern_weights['late_night'] +
                    merchant_scores[merchant_id]['velocity']['velocity_ratio'] * pattern_weights['velocity'] +
                    merchant_scores[merchant_id]['split_transactions']['split_transaction_ratio'] * pattern_weights['split_transactions'] +
                    merchant_scores[merchant_id]['round_amounts']['round_amount_ratio'] * pattern_weights['round_amounts'] +
                    merchant_scores[merchant_id]['customer_concentration']['concentration_ratio'] * pattern_weights['customer_concentration']
                )
                
                merchant_scores[merchant_id]['overall_risk_score'] = risk_score

            except Exception as e:
                logging.error(f"Error processing merchant {merchant_id}: {str(e)}")
                continue

        return merchant_scores

    def get_high_risk_merchants(self, merchant_scores: Dict[str, Dict], 
                              risk_threshold: float = 0.5) -> List[str]:
        """Identify high-risk merchants with adjusted threshold"""
        return [
            merchant_id for merchant_id, scores in merchant_scores.items()
            if scores.get('overall_risk_score', 0) > risk_threshold
        ] 