import logging
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict
import os

from data_generation import DataGenerator
from feature_engineering import FeatureEngineering
from anomaly_detection import AutoencoderModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudDetectionPipeline:
    def __init__(self, data_path: Path = Path('../data')):
        self.data_path = data_path
        self.data_generator = DataGenerator(data_path)
        self.feature_engineering = FeatureEngineering()
        self.model = None
        
    def generate_datasets(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate both training and test datasets"""
        # Generate training data (normal merchants only)
        train_merchants, train_transactions = self.data_generator.generate_training_data(
            merchant_count=150
        )
        logger.info(f"Generated training data: {len(train_merchants)} merchants, {len(train_transactions)} transactions")
        
        # Generate test data (mix of normal and fraudulent merchants)
        test_merchants, test_transactions = self.data_generator.generate_test_data(
            merchant_count=100,
            fraud_percentage=0.5
        )
        logger.info(f"Generated test data: {len(test_merchants)} merchants, {len(test_transactions)} transactions")
        
        return train_merchants, train_transactions, test_merchants, test_transactions
    
    def train_model(self, train_transactions: pd.DataFrame) -> None:
        """Train the anomaly detection model"""
        # Create features for training data
        logger.info("Creating features for training data...")
        train_features = self.feature_engineering.create_merchant_features(train_transactions)
        
        # Prepare training data
        X_train = self.feature_engineering.prepare_training_data(train_features)
        
        # Initialize and train model
        self.model = AutoencoderModel(input_dim=X_train.shape[1])
        self.model.train(X_train, epochs=40, batch_size=32)
        
        # Plot training results
        self._plot_training_metrics(train_features)
    
    def detect_anomalies(self, test_transactions: pd.DataFrame, 
                        test_merchants: pd.DataFrame) -> Dict:
        """Detect anomalies in test data"""
        if self.model is None:
            raise ValueError("Model not trained. Run train_model() first.")
        
        # Create features for test data
        logger.info("Creating features for test data...")
        test_features = self.feature_engineering.create_merchant_features(test_transactions)
        
        # Prepare test data
        X_test = self.feature_engineering.prepare_test_data(test_features)
        
        # Get predictions and scores
        predictions, anomaly_scores = self.model.predict_anomalies(X_test)
        
        # Combine results
        results = {
            'merchant_id': test_features['merchant_id'],
            'is_anomaly': predictions,
            'anomaly_score': anomaly_scores
        }
        
        # Calculate metrics
        metrics = self._calculate_detection_metrics(
            test_merchants,
            predictions,
            anomaly_scores
        )
        
        # Plot results
        self._plot_detection_results(results, test_merchants)
        
        return {'predictions': results, 'metrics': metrics}
    
    def _calculate_detection_metrics(self, test_merchants: pd.DataFrame,
                                  predictions: np.ndarray,
                                  scores: np.ndarray) -> Dict:
        """Calculate detection performance metrics"""
        true_labels = test_merchants['is_fraud'].values
        
        metrics = {
            'accuracy': np.mean(predictions == true_labels),
            'precision': np.sum((predictions == 1) & (true_labels == 1)) / np.sum(predictions == 1),
            'recall': np.sum((predictions == 1) & (true_labels == 1)) / np.sum(true_labels == 1),
            'avg_anomaly_score': np.mean(scores),
            'fraud_detection_rate': np.mean(scores[true_labels == 1]),
            'false_positive_rate': np.mean(predictions[true_labels == 0] == 1)
        }
        
        return metrics
    
    def _plot_training_metrics(self, train_features: pd.DataFrame) -> None:
        """Plot training data distributions"""
        plt.figure(figsize=(15, 10))
        
        # Plot feature distributions
        for i, col in enumerate(train_features.columns[1:], 1):
            plt.subplot(3, 5, i)
            sns.histplot(train_features[col], bins=30)
            plt.title(col)
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.data_path / 'training_metrics.png')
        plt.close()
    
    def _plot_detection_results(self, results: Dict, 
                              test_merchants: pd.DataFrame) -> None:
        """Plot anomaly detection results"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Score Distribution
        plt.subplot(131)
        sns.histplot(results['anomaly_score'], bins=30)
        plt.title('Anomaly Score Distribution')
        
        # Plot 2: Score by Merchant Type
        plt.subplot(132)
        sns.boxplot(x='is_fraud', y='anomaly_score', data={
            'is_fraud': test_merchants['is_fraud'],
            'anomaly_score': results['anomaly_score']
        })
        plt.title('Scores by Merchant Type')
        
        # Plot 3: ROC-like curve
        plt.subplot(133)
        thresholds = np.linspace(0, 1, 100)
        tpr = [np.mean(results['anomaly_score'][test_merchants['is_fraud']] > t) for t in thresholds]
        fpr = [np.mean(results['anomaly_score'][~test_merchants['is_fraud']] > t) for t in thresholds]
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1], '--')
        plt.title('Detection Performance')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        
        plt.tight_layout()
        plt.savefig(self.data_path / 'detection_results.png')
        plt.close()
    
    def validate_data(self, train_transactions: pd.DataFrame, test_transactions: pd.DataFrame,
                      train_merchants: pd.DataFrame, test_merchants: pd.DataFrame) -> Dict:
        """Validate datasets against multiple scenarios"""
        validation_results = {
            'fraud_pattern_validation': self._validate_fraud_patterns(test_transactions, test_merchants),
            'dataset_balance': self._validate_dataset_balance(train_merchants, test_merchants)
        }
        
        # Generate fraud report explicitly
        fraud_merchants = test_merchants[test_merchants['is_fraud']]
        fraud_transactions = test_transactions[
            test_transactions['merchant_id'].isin(fraud_merchants['merchant_id'])
        ]
        self._generate_fraud_report(fraud_transactions, fraud_merchants)
        
        # Log validation results
        logger.info("Data Validation Results:")
        for category, results in validation_results.items():
            logger.info(f"\n{category.upper()}:")
            for metric, value in results.items():
                logger.info(f"{metric}: {value}")
        
        return validation_results

    def _validate_fraud_patterns(self, transactions: pd.DataFrame, 
                                 merchants: pd.DataFrame) -> Dict:
        """Validate fraud pattern characteristics"""
        results = {}
        
        # Merge transactions with merchant data to identify fraudulent transactions
        merged_data = transactions.merge(merchants[['merchant_id', 'is_fraud']], 
                                         on='merchant_id', how='left')
        fraud_trans = merged_data[merged_data['is_fraud']]
        normal_trans = merged_data[~merged_data['is_fraud']]
        
        # Pattern Characteristics
        results['pattern_characteristics'] = {
            'fraud_avg_amount': fraud_trans['amount'].mean(),
            'normal_avg_amount': normal_trans['amount'].mean(),
            'fraud_amount_std': fraud_trans['amount'].std(),
            'normal_amount_std': normal_trans['amount'].std()
        }
        
        # Pattern Timing
        fraud_trans.loc[:, 'hour'] = pd.to_datetime(fraud_trans['timestamp']).dt.hour
        results['pattern_timing'] = {
            'off_hours_ratio': len(fraud_trans[~fraud_trans['hour'].between(9, 17)]) / len(fraud_trans),
            'hour_concentration': fraud_trans['hour'].value_counts().std()
        }
        
        # Pattern Intensity
        fraud_merchant_activity = fraud_trans.groupby('merchant_id').agg({
            'amount': ['count', 'sum', 'mean']
        })
        results['pattern_intensity'] = {
            'avg_transactions_per_fraud_merchant': fraud_merchant_activity[('amount', 'count')].mean(),
            'avg_amount_per_fraud_merchant': fraud_merchant_activity[('amount', 'sum')].mean()
        }
        
        return results

    def _validate_dataset_balance(self, train_merchants: pd.DataFrame, 
                                  test_merchants: pd.DataFrame) -> Dict:
        """Validate dataset balance and statistics"""
        results = {}
        
        # Fraud/Normal Ratio
        test_fraud_ratio = test_merchants['is_fraud'].mean()
        results['fraud_ratio'] = {
            'test_set_fraud_ratio': test_fraud_ratio,
            'train_set_fraud_ratio': train_merchants['is_fraud'].mean() if 'is_fraud' in train_merchants.columns else 0
        }
        
        # Pattern Distribution
        if 'merchant_type' in test_merchants.columns:
            results['pattern_distribution'] = {
                'merchant_type_distribution': test_merchants['merchant_type'].value_counts().to_dict()
            }
        
        # Overall Statistics
        results['overall_statistics'] = {
            'train_merchant_count': len(train_merchants),
            'test_merchant_count': len(test_merchants),
            'test_fraud_merchant_count': test_merchants['is_fraud'].sum(),
            'test_normal_merchant_count': len(test_merchants) - test_merchants['is_fraud'].sum()
        }
        
        return results

    def _generate_fraud_report(self, fraud_transactions: pd.DataFrame, 
                               fraud_merchants: pd.DataFrame) -> None:
        """Generate detailed report of fraudulent merchants and a summary list"""
        # Ensure the data directory exists
        try:
            os.makedirs(self.data_path, exist_ok=True)
            logging.info(f"Data directory created at: {self.data_path}")
        except Exception as e:
            logging.error(f"Failed to create data directory: {e}")
            return
        
        report_path = self.data_path / 'fraud_merchant_report.txt'
        summary_path = self.data_path / 'fraud_merchant_summary.txt'
        
        try:
            with open(report_path, 'w') as report_file, open(summary_path, 'w') as summary_file:
                logging.info(f"Writing report to: {report_path}")
                logging.info(f"Writing summary to: {summary_path}")
                
                # Write summary header
                summary_file.write("=== FRAUD MERCHANT SUMMARY LIST ===\n")
                summary_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                summary_file.write("Merchant ID | Total Transactions | Avg Amount | Max Amount | Off-hours Ratio | Fraud Type\n")
                summary_file.write("-" * 100 + "\n")
                
                # Write detailed report header
                report_file.write("=== FRAUD MERCHANT ANALYSIS REPORT ===\n")
                report_file.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                report_file.write(f"Total Fraud Merchants: {len(fraud_merchants)}\n")
                report_file.write(f"Total Fraudulent Transactions: {len(fraud_transactions)}\n\n")
                
                # Process each fraudulent merchant
                for _, merchant in fraud_merchants.iterrows():
                    merchant_id = merchant['merchant_id']
                    merchant_trans = fraud_transactions[fraud_transactions['merchant_id'] == merchant_id]
                    
                    if len(merchant_trans) == 0:
                        logging.warning(f"No transactions found for merchant {merchant_id}")
                        continue
                    
                    # Calculate metrics
                    trans_hours = pd.to_datetime(merchant_trans['timestamp']).dt.hour
                    off_hours_ratio = len(trans_hours[~trans_hours.between(9, 17)]) / len(trans_hours)
                    avg_amount = merchant_trans['amount'].mean()
                    max_amount = merchant_trans['amount'].max()
                    unique_customers = merchant_trans['customer_id'].nunique()
                    customer_concentration = (
                        merchant_trans['customer_id'].value_counts().max() / len(merchant_trans)
                    )
                    
                    # Classify fraud type
                    fraud_type = self._classify_fraud_type(
                        merchant_trans,
                        off_hours_ratio,
                        customer_concentration
                    )
                    
                    # Write to summary file
                    summary_line = (
                        f"{merchant_id} | "
                        f"{len(merchant_trans)} | "
                        f"${avg_amount:.2f} | "
                        f"${max_amount:.2f} | "
                        f"{off_hours_ratio:.1%} | "
                        f"{fraud_type}\n"
                    )
                    summary_file.write(summary_line)
                    
                    # Write detailed report
                    report_file.write(f"\nMERCHANT ID: {merchant_id}\n")
                    report_file.write("-" * 50 + "\n")
                    report_file.write("\nTransaction Patterns:\n")
                    report_file.write(f"- Total Transactions: {len(merchant_trans)}\n")
                    report_file.write(f"- Average Amount: ${avg_amount:.2f}\n")
                    report_file.write(f"- Maximum Amount: ${max_amount:.2f}\n")
                    report_file.write(f"- Off-hours Transaction Ratio: {off_hours_ratio:.1%}\n")
                    report_file.write("\nCustomer Analysis:\n")
                    report_file.write(f"- Unique Customers: {unique_customers}\n")
                    report_file.write(f"- Customer Concentration: {customer_concentration:.1%}\n")
                    report_file.write(f"\nSuspected Fraud Type: {fraud_type}\n")
                    report_file.write("-" * 50 + "\n")
                
                # Write summary statistics
                report_file.write("\n=== SUMMARY STATISTICS ===\n")
                report_file.write(f"Total Fraudulent Merchants: {len(fraud_merchants)}\n")
                report_file.write(f"Average Transactions per Merchant: {len(fraud_transactions)/len(fraud_merchants):.1f}\n")
                
                logging.info("Successfully generated fraud reports")
                
        except Exception as e:
            logging.error(f"Failed to write report or summary: {e}")
            raise

    def _classify_fraud_type(self, transactions: pd.DataFrame, 
                            off_hours_ratio: float,
                            customer_concentration: float) -> str:
        """Classify the type of fraud based on transaction patterns"""
        avg_amount = transactions['amount'].mean()
        trans_count = len(transactions)
        
        fraud_indicators = []
        
        # Check for high-value fraud
        if avg_amount > 1000:
            fraud_indicators.append("High-value transaction fraud")
        
        # Check for velocity fraud (high number of transactions)
        if trans_count > 100:
            fraud_indicators.append("High-velocity fraud")
        
        # Check for off-hours fraud
        if off_hours_ratio > 0.7:
            fraud_indicators.append("Off-hours operation")
        
        # Check for customer pattern fraud
        if customer_concentration > 0.8:
            fraud_indicators.append("Customer concentration fraud")
        elif transactions['customer_id'].nunique() < 5:
            fraud_indicators.append("Limited customer base")
        
        if not fraud_indicators:
            return "Unknown pattern"
        
        return " + ".join(fraud_indicators)

def main():
    # Initialize pipeline
    pipeline = FraudDetectionPipeline()
    
    # Generate datasets
    train_merchants, train_transactions, test_merchants, test_transactions = pipeline.generate_datasets()
    
    # Validate datasets
    validation_results = pipeline.validate_data(
        train_transactions=train_transactions,
        test_transactions=test_transactions,
        train_merchants=train_merchants,
        test_merchants=test_merchants
    )
    
    # Train model
    pipeline.train_model(train_transactions)
    
    # Detect anomalies
    results = pipeline.detect_anomalies(test_transactions, test_merchants)
    
    # Log results
    logger.info("Detection Metrics:")
    for metric, value in results['metrics'].items():
        logger.info(f"{metric}: {value:.4f}")

if __name__ == "__main__":
    main() 