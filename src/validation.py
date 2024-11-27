import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

class DatasetValidator:
    def __init__(self, merchants_df: pd.DataFrame, transactions_df: pd.DataFrame):
        self.merchants_df = merchants_df
        self.transactions_df = transactions_df.copy()
        self.transactions_df['timestamp'] = pd.to_datetime(self.transactions_df['timestamp'])
        self.transactions_df['hour'] = self.transactions_df['timestamp'].dt.hour
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def validate_normal_merchants(self) -> Dict:
        """Validate normal merchant behavior patterns"""
        fraud_merchants = self.get_fraud_merchants()
        normal_merchants = self.transactions_df[~self.transactions_df['merchant_id'].isin(fraud_merchants)]
        
        results = {
            'business_hours': self._check_business_hours(normal_merchants),
            'amount_distribution': self._verify_amount_distribution(normal_merchants),
            'customer_diversity': self._validate_customer_diversity(normal_merchants)
        }
        
        self._plot_normal_merchant_metrics(results)
        return results

    def validate_fraud_patterns(self) -> Dict:
        """Validate fraud pattern characteristics"""
        fraud_merchants = self.get_fraud_merchants()
        fraud_txns = self.transactions_df[self.transactions_df['merchant_id'].isin(fraud_merchants)]
        
        results = {
            'pattern_characteristics': self._verify_pattern_characteristics(fraud_txns),
            'pattern_timing': self._check_pattern_timing(fraud_txns),
            'pattern_intensity': self._validate_pattern_intensity(fraud_txns)
        }
        
        self._plot_fraud_pattern_metrics(results)
        return results

    def validate_dataset_balance(self) -> Dict:
        """Validate overall dataset statistics and balance"""
        results = {
            'fraud_ratio': self._check_fraud_ratio(),
            'pattern_distribution': self._verify_pattern_distribution(),
            'overall_statistics': self._validate_overall_statistics()
        }
        
        self._plot_dataset_balance_metrics(results)
        return results

    def _check_business_hours(self, txns: pd.DataFrame) -> Dict:
        """Check if normal merchants operate primarily during business hours"""
        hour_dist = txns['hour'].value_counts().sort_index()
        business_hours_txns = txns[(txns['hour'] >= 9) & (txns['hour'] <= 18)]
        
        return {
            'business_hours_ratio': len(business_hours_txns) / len(txns),
            'hour_distribution': hour_dist.to_dict(),
            'peak_hour': hour_dist.idxmax(),
            'off_hours_percentage': len(txns[~txns['hour'].between(9, 18)]) / len(txns)
        }

    def _verify_amount_distribution(self, txns: pd.DataFrame) -> Dict:
        """Verify transaction amount distributions"""
        return {
            'mean_amount': txns['amount'].mean(),
            'median_amount': txns['amount'].median(),
            'std_amount': txns['amount'].std(),
            'amount_percentiles': txns['amount'].quantile([0.25, 0.5, 0.75]).to_dict()
        }

    def _validate_customer_diversity(self, txns: pd.DataFrame) -> Dict:
        """Check customer diversity metrics"""
        customer_counts = txns.groupby('merchant_id')['customer_id'].nunique()
        return {
            'avg_customers_per_merchant': customer_counts.mean(),
            'customer_concentration': (txns.groupby('merchant_id')['customer_id']
                                    .value_counts()
                                    .groupby(level=0)
                                    .head(1)
                                    .mean()),
            'customer_diversity_score': customer_counts.std() / customer_counts.mean()
        }

    def _verify_pattern_characteristics(self, txns: pd.DataFrame) -> Dict:
        """Verify specific fraud pattern characteristics"""
        patterns = {
            'late_night': self._check_late_night_pattern(txns),
            'velocity': self._check_velocity_pattern(txns),
            'split': self._check_split_pattern(txns),
            'round_amount': self._check_round_amount_pattern(txns),
            'customer_concentration': self._check_customer_concentration_pattern(txns)
        }
        return patterns

    def _check_pattern_timing(self, txns: pd.DataFrame) -> Dict:
        """Analyze timing aspects of fraud patterns"""
        return {
            'pattern_duration': self._calculate_pattern_duration(txns),
            'pattern_frequency': self._calculate_pattern_frequency(txns),
            'temporal_distribution': self._analyze_temporal_distribution(txns)
        }

    def _validate_pattern_intensity(self, txns: pd.DataFrame) -> Dict:
        """Measure the intensity of fraud patterns"""
        return {
            'transaction_volume': self._analyze_transaction_volume(txns),
            'amount_intensity': self._analyze_amount_intensity(txns),
            'pattern_consistency': self._measure_pattern_consistency(txns)
        }

    def get_fraud_merchants(self) -> List[str]:
        """Identify merchants with fraud patterns based on transaction patterns"""
        fraud_indicators = []
        
        for merchant_id in self.transactions_df['merchant_id'].unique():
            merchant_txns = self.transactions_df[self.transactions_df['merchant_id'] == merchant_id]
            
            # Check for various fraud indicators
            indicators = {
                # Late night pattern
                'late_night': (merchant_txns['hour'].isin([23, 0, 1, 2, 3, 4]).mean() > 0.3),
                
                # Velocity pattern
                'high_velocity': (merchant_txns.groupby(merchant_txns['timestamp'].dt.date).size().max() > 200),
                
                # Split transaction pattern
                'split_txn': (
                    merchant_txns.groupby(['customer_id', merchant_txns['timestamp'].dt.date])
                    .size().max() > 8
                ),
                
                # Round amount pattern
                'round_amount': (
                    merchant_txns['amount'].astype(str)
                    .str.endswith(('999', '000')).mean() > 0.3
                ),
                
                # Customer concentration pattern
                'customer_concentration': (
                    merchant_txns['customer_id'].value_counts().head(5).sum() / 
                    len(merchant_txns) > 0.7
                )
            }
            
            # If any fraud indicator is present, add to fraud merchants list
            if any(indicators.values()):
                fraud_indicators.append(merchant_id)
        
        self.logger.info(f"Identified {len(fraud_indicators)} merchants with fraud patterns")
        return fraud_indicators

    def _plot_normal_merchant_metrics(self, results: Dict):
        """Plot normal merchant validation metrics"""
        plt.figure(figsize=(15, 5))
        
        # Plot business hours distribution
        plt.subplot(131)
        hours = list(results['business_hours']['hour_distribution'].keys())
        counts = list(results['business_hours']['hour_distribution'].values())
        plt.bar(hours, counts)
        plt.title('Business Hours Distribution')
        plt.xlabel('Hour of Day')
        plt.ylabel('Transaction Count')
        
        # Plot amount distribution
        plt.subplot(132)
        sns.histplot(self.transactions_df['amount'], bins=50)
        plt.title('Amount Distribution')
        plt.xlabel('Transaction Amount')
        
        # Plot customer diversity
        plt.subplot(133)
        customer_counts = self.transactions_df.groupby('merchant_id')['customer_id'].nunique()
        sns.histplot(customer_counts, bins=30)
        plt.title('Customer Diversity')
        plt.xlabel('Unique Customers per Merchant')
        
        plt.tight_layout()
        plt.savefig('data/normal_merchant_validation.png')
        plt.close()

    def _plot_fraud_pattern_metrics(self, results: Dict):
        """Plot fraud pattern validation metrics"""
        plt.figure(figsize=(20, 10))
        
        # Plot 1: Pattern Characteristics
        plt.subplot(231)
        pattern_chars = results['pattern_characteristics']
        metrics = {
            'Late Night': pattern_chars['late_night']['late_night_ratio'],
            'Velocity': pattern_chars['velocity']['spike_ratio'] / 10,  # Normalized
            'Split Trans.': pattern_chars['split']['split_ratio'],
            'Round Amount': pattern_chars['round_amount']['round_amount_ratio'],
            'Cust. Conc.': pattern_chars['customer_concentration']['top_5_customer_ratio']
        }
        plt.bar(metrics.keys(), metrics.values())
        plt.title('Fraud Pattern Characteristics')
        plt.xticks(rotation=45)
        
        # Plot 2: Temporal Distribution
        plt.subplot(232)
        hour_dist = results['pattern_timing']['temporal_distribution']['hour_distribution']
        plt.plot(list(hour_dist.keys()), list(hour_dist.values()), marker='o')
        plt.title('Hourly Transaction Distribution')
        plt.xlabel('Hour of Day')
        plt.ylabel('Transaction Ratio')
        
        # Plot 3: Pattern Intensity
        plt.subplot(233)
        volume_data = results['pattern_intensity']['transaction_volume']
        plt.bar(['Total', 'Daily Avg', 'Daily Std'], 
                [volume_data['total_volume'], 
                 volume_data['daily_mean'], 
                 volume_data['daily_std']])
        plt.title('Transaction Volume Metrics')
        
        # Plot 4: Amount Distribution
        plt.subplot(234)
        amount_data = results['pattern_intensity']['amount_intensity']
        plt.bar(['Mean Amount', 'Amount Std'], 
                [amount_data['mean_amount'], 
                 amount_data['amount_std']])
        plt.title('Amount Distribution')
        
        # Plot 5: Pattern Consistency
        plt.subplot(235)
        consistency = results['pattern_intensity']['pattern_consistency']
        plt.bar(['Amount Consistency', 'Pattern Stability'], 
                [consistency['amount_consistency'], 
                 consistency['pattern_stability']])
        plt.title('Pattern Consistency Metrics')
        
        plt.tight_layout()
        plt.savefig('data/fraud_pattern_validation.png')
        plt.close()

    def _plot_dataset_balance_metrics(self, results: Dict):
        """Plot dataset balance metrics"""
        plt.figure(figsize=(20, 10))
        
        # Plot 1: Fraud Ratios
        plt.subplot(231)
        fraud_ratios = results['fraud_ratio']
        plt.bar(['Merchant Ratio', 'Transaction Ratio'], 
                [fraud_ratios['fraud_merchant_ratio'], 
                 fraud_ratios['fraud_transaction_ratio']])
        plt.title('Fraud Ratios')
        plt.ylabel('Ratio')
        
        # Plot 2: Pattern Distribution
        plt.subplot(232)
        pattern_dist = results['pattern_distribution']['pattern_counts']
        plt.bar(pattern_dist.keys(), pattern_dist.values())
        plt.title('Fraud Pattern Distribution')
        plt.xticks(rotation=45)
        
        # Plot 3: Transaction Stats
        plt.subplot(233)
        txn_stats = results['overall_statistics']['transaction_stats']
        plt.bar(['Transactions/Day', 'Avg Amount (scaled)'], 
                [txn_stats['transactions_per_day'], 
                 txn_stats['avg_transaction_amount']/1000])
        plt.title('Transaction Statistics')
        
        # Plot 4: Merchant Stats
        plt.subplot(234)
        merchant_stats = results['overall_statistics']['merchant_stats']
        plt.bar(['Txns/Merchant', 'Activity Days'], 
                [merchant_stats['avg_transactions_per_merchant'],
                 merchant_stats['merchant_activity_days']])
        plt.title('Merchant Statistics')
        
        # Plot 5: Customer Stats
        plt.subplot(235)
        customer_stats = results['overall_statistics']['customer_stats']
        plt.bar(['Customers/Merchant', 'Txns/Customer'], 
                [customer_stats['avg_customers_per_merchant'],
                 customer_stats['transactions_per_customer']])
        plt.title('Customer Statistics')
        
        # Plot 6: Temporal Stats
        plt.subplot(236)
        temporal_stats = results['overall_statistics']['temporal_stats']
        plt.bar(['Business Hours', 'Weekend Ratio'], 
                [temporal_stats['business_hours_ratio'],
                 temporal_stats['weekend_ratio']])
        plt.title('Temporal Distribution')
        
        plt.tight_layout()
        plt.savefig('data/dataset_balance_validation.png')
        plt.close()

    def _check_late_night_pattern(self, txns: pd.DataFrame) -> Dict:
        """Analyze late night transaction patterns"""
        late_night_hours = [23, 0, 1, 2, 3, 4]
        late_night_txns = txns[txns['hour'].isin(late_night_hours)]
        
        return {
            'late_night_ratio': len(late_night_txns) / len(txns) if len(txns) > 0 else 0,
            'avg_late_night_amount': late_night_txns['amount'].mean() if len(late_night_txns) > 0 else 0,
            'late_night_volume': len(late_night_txns),
            'peak_late_night_hour': late_night_txns['hour'].mode().iloc[0] if len(late_night_txns) > 0 else None
        }

    def _check_velocity_pattern(self, txns: pd.DataFrame) -> Dict:
        """Analyze transaction velocity patterns"""
        daily_volumes = txns.groupby(txns['timestamp'].dt.date).size()
        return {
            'max_daily_volume': daily_volumes.max(),
            'avg_daily_volume': daily_volumes.mean(),
            'volume_std': daily_volumes.std(),
            'spike_ratio': daily_volumes.max() / daily_volumes.mean() if daily_volumes.mean() > 0 else 0
        }

    def _check_split_pattern(self, txns: pd.DataFrame) -> Dict:
        """Analyze split transaction patterns"""
        if len(txns) == 0:
            return {
                'split_ratio': 0,
                'avg_split_amount': 0,
                'split_customer_count': 0,
                'max_splits_per_customer': 0
            }
        
        # Convert to datetime if not already
        txns = txns.sort_values('timestamp')
        
        # Calculate time differences for each customer
        time_diffs = pd.Series(dtype='timedelta64[ns]')
        for customer in txns['customer_id'].unique():
            customer_txns = txns[txns['customer_id'] == customer]
            diffs = customer_txns['timestamp'].diff()
            time_diffs = pd.concat([time_diffs, diffs])
        
        # Identify split transactions (within 30 minutes)
        split_mask = time_diffs <= pd.Timedelta(minutes=30)
        split_candidates = txns[split_mask.fillna(False)]
        
        return {
            'split_ratio': len(split_candidates) / len(txns),
            'avg_split_amount': split_candidates['amount'].mean() if len(split_candidates) > 0 else 0,
            'split_customer_count': split_candidates['customer_id'].nunique(),
            'max_splits_per_customer': split_candidates.groupby('customer_id').size().max() if len(split_candidates) > 0 else 0
        }

    def _check_round_amount_pattern(self, txns: pd.DataFrame) -> Dict:
        """Analyze round amount patterns"""
        amounts = txns['amount']
        round_amounts = amounts[amounts.astype(str).str.endswith(('999', '000'))]
        
        return {
            'round_amount_ratio': len(round_amounts) / len(amounts) if len(amounts) > 0 else 0,
            'avg_round_amount': round_amounts.mean() if len(round_amounts) > 0 else 0,
            'round_amount_count': len(round_amounts),
            'most_common_round_amount': round_amounts.mode().iloc[0] if len(round_amounts) > 0 else None
        }

    def _check_customer_concentration_pattern(self, txns: pd.DataFrame) -> Dict:
        """Analyze customer concentration patterns"""
        customer_txn_counts = txns['customer_id'].value_counts()
        top_5_concentration = customer_txn_counts.head(5).sum() / len(txns) if len(txns) > 0 else 0
        
        return {
            'top_5_customer_ratio': top_5_concentration,
            'unique_customer_ratio': len(customer_txn_counts) / len(txns) if len(txns) > 0 else 0,
            'max_customer_ratio': customer_txn_counts.max() / len(txns) if len(txns) > 0 else 0,
            'avg_txns_per_customer': len(txns) / len(customer_txn_counts) if len(customer_txn_counts) > 0 else 0
        }

    def _calculate_pattern_duration(self, txns: pd.DataFrame) -> Dict:
        """Calculate duration of pattern activity"""
        return {
            'start_date': txns['timestamp'].min(),
            'end_date': txns['timestamp'].max(),
            'duration_days': (txns['timestamp'].max() - txns['timestamp'].min()).days,
            'active_days': txns['timestamp'].dt.date.nunique()
        }

    def _calculate_pattern_frequency(self, txns: pd.DataFrame) -> Dict:
        """Calculate frequency of pattern occurrence"""
        daily_patterns = txns.groupby(txns['timestamp'].dt.date).size()
        return {
            'daily_avg': daily_patterns.mean(),
            'daily_std': daily_patterns.std(),
            'max_daily': daily_patterns.max(),
            'min_daily': daily_patterns.min()
        }

    def _analyze_temporal_distribution(self, txns: pd.DataFrame) -> Dict:
        """Analyze temporal distribution of patterns"""
        hourly_dist = txns['hour'].value_counts(normalize=True).sort_index()
        return {
            'peak_hours': hourly_dist.nlargest(3).index.tolist(),
            'hour_distribution': hourly_dist.to_dict(),
            'weekend_ratio': txns[txns['timestamp'].dt.dayofweek.isin([5, 6])].shape[0] / len(txns)
        }

    def _analyze_transaction_volume(self, txns: pd.DataFrame) -> Dict:
        """Analyze transaction volume patterns"""
        daily_volumes = txns.groupby(txns['timestamp'].dt.date).size()
        return {
            'total_volume': len(txns),
            'daily_mean': daily_volumes.mean(),
            'daily_std': daily_volumes.std(),
            'volume_trend': daily_volumes.pct_change().mean()
        }

    def _analyze_amount_intensity(self, txns: pd.DataFrame) -> Dict:
        """Analyze transaction amount intensity"""
        return {
            'total_amount': txns['amount'].sum(),
            'mean_amount': txns['amount'].mean(),
            'amount_std': txns['amount'].std(),
            'amount_trend': txns.groupby(txns['timestamp'].dt.date)['amount'].mean().pct_change().mean()
        }

    def _measure_pattern_consistency(self, txns: pd.DataFrame) -> Dict:
        """Measure consistency of pattern behavior"""
        daily_stats = txns.groupby(txns['timestamp'].dt.date).agg({
            'amount': ['mean', 'std'],
            'customer_id': 'nunique'
        })
        
        return {
            'amount_consistency': daily_stats['amount']['std'].std(),
            'volume_consistency': daily_stats['customer_id'].std(),
            'pattern_stability': daily_stats['amount']['mean'].pct_change().std()
        }

    def _check_fraud_ratio(self) -> Dict:
        """Check the ratio of fraudulent to normal transactions"""
        fraud_merchants = self.get_fraud_merchants()
        fraud_txns = self.transactions_df[self.transactions_df['merchant_id'].isin(fraud_merchants)]
        
        return {
            'fraud_merchant_ratio': len(fraud_merchants) / len(self.merchants_df),
            'fraud_transaction_ratio': len(fraud_txns) / len(self.transactions_df),
            'fraud_merchant_count': len(fraud_merchants),
            'total_merchant_count': len(self.merchants_df),
            'fraud_transaction_count': len(fraud_txns),
            'total_transaction_count': len(self.transactions_df)
        }

    def _verify_pattern_distribution(self) -> Dict:
        """Verify the distribution of different fraud patterns"""
        fraud_merchants = self.get_fraud_merchants()
        pattern_counts = {
            'late_night': 0,
            'velocity': 0,
            'split': 0,
            'round_amount': 0,
            'customer_concentration': 0
        }
        
        # Check patterns for each fraud merchant individually
        for merchant_id in fraud_merchants:
            merchant_txns = self.transactions_df[self.transactions_df['merchant_id'] == merchant_id]
            
            # Check each pattern type
            if self._check_late_night_pattern(merchant_txns)['late_night_ratio'] > 0.3:
                pattern_counts['late_night'] += 1
                
            if self._check_velocity_pattern(merchant_txns)['spike_ratio'] > 4.0:
                pattern_counts['velocity'] += 1
                
            if self._check_split_pattern(merchant_txns)['split_ratio'] > 0.2:
                pattern_counts['split'] += 1
                
            if self._check_round_amount_pattern(merchant_txns)['round_amount_ratio'] > 0.3:
                pattern_counts['round_amount'] += 1
                
            if self._check_customer_concentration_pattern(merchant_txns)['top_5_customer_ratio'] > 0.7:
                pattern_counts['customer_concentration'] += 1
        
        total_patterns = sum(pattern_counts.values())
        
        return {
            'pattern_counts': pattern_counts,
            'pattern_distribution': {
                k: v/total_patterns if total_patterns > 0 else 0 
                for k, v in pattern_counts.items()
            },
            'total_patterns': total_patterns,
            'patterns_per_merchant': total_patterns / len(fraud_merchants) if len(fraud_merchants) > 0 else 0,
            'merchant_count': len(fraud_merchants)
        }

    def _validate_overall_statistics(self) -> Dict:
        """Validate overall dataset statistics"""
        # Calculate merchant activity days correctly
        merchant_activity = self.transactions_df.groupby('merchant_id').agg({
            'timestamp': lambda x: x.dt.date.nunique()
        })
        
        return {
            'transaction_stats': {
                'total_transactions': len(self.transactions_df),
                'transactions_per_day': len(self.transactions_df) / self.transactions_df['timestamp'].dt.date.nunique(),
                'avg_transaction_amount': self.transactions_df['amount'].mean(),
                'std_transaction_amount': self.transactions_df['amount'].std()
            },
            'merchant_stats': {
                'total_merchants': len(self.merchants_df),
                'avg_transactions_per_merchant': len(self.transactions_df) / len(self.merchants_df),
                'merchant_activity_days': merchant_activity['timestamp'].mean()
            },
            'customer_stats': {
                'total_customers': self.transactions_df['customer_id'].nunique(),
                'avg_customers_per_merchant': self.transactions_df.groupby('merchant_id')['customer_id'].nunique().mean(),
                'transactions_per_customer': len(self.transactions_df) / self.transactions_df['customer_id'].nunique()
            },
            'temporal_stats': {
                'dataset_duration_days': (self.transactions_df['timestamp'].max() - 
                                        self.transactions_df['timestamp'].min()).days,
                'business_hours_ratio': len(self.transactions_df[self.transactions_df['hour'].between(9, 18)]) / 
                                      len(self.transactions_df),
                'weekend_ratio': len(self.transactions_df[self.transactions_df['timestamp'].dt.dayofweek.isin([5, 6])]) / 
                             len(self.transactions_df)
            }
        }
  