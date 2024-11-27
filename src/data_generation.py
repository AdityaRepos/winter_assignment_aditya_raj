import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import pandas as pd
from pathlib import Path
import logging

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Constants
BUSINESS_TYPES = ['Electronics', 'Fashion', 'Grocery', 'Restaurant', 'Services']
BUSINESS_MODELS = ['Online', 'Offline', 'Hybrid']
PRODUCT_CATEGORIES = ['Electronics', 'Clothing', 'Food', 'Services', 'Home Goods']
CITIES = ['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata']
STATES = ['Maharashtra', 'Delhi', 'Karnataka', 'Tamil Nadu', 'West Bengal']
PAYMENT_METHODS = ['Credit Card', 'Debit Card', 'UPI', 'Net Banking']
PLATFORMS = ['Web', 'Mobile', 'POS']

class DataGenerator:
    def __init__(self, data_path: Path = Path('data')):
        self.data_path = data_path
        self.data_path.mkdir(exist_ok=True)
        
        # Merchant profile configurations
        self.business_types = ['retail', 'restaurant', 'service', 'online', 'wholesale']
        self.gst_statuses = ['registered', 'unregistered']
        self.volume_categories = ['low', 'medium', 'high']
        
        self.start_date = datetime(2020, 1, 1)
        self.end_date = datetime.now()

    def generate_training_data(self, merchant_count: int = 1000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate clean training data with only normal merchant behavior"""
        logger.info("Generating training dataset with normal merchant behavior...")
        
        # Generate merchant profiles
        merchants = []
        for _ in range(merchant_count):
            merchant = {
                'merchant_id': f"M{uuid.uuid4().hex[:8].upper()}",
                'business_type': random.choice(self.business_types),
                'registration_date': self._generate_registration_date(),
                'gst_status': random.choice(self.gst_statuses),
                'volume_category': random.choice(self.volume_categories)
            }
            merchants.append(merchant)
        
        # Generate normal transactions
        transactions = []
        for merchant in merchants:
            merchant_txns = self._generate_normal_transactions(
                merchant['merchant_id'],
                merchant['volume_category'],
                days=30
            )
            transactions.extend(merchant_txns)
        
        merchants_df = pd.DataFrame(merchants)
        transactions_df = pd.DataFrame(transactions)
        
        # Save training data
        merchants_df.to_csv(self.data_path / 'training_merchants.csv', index=False)
        transactions_df.to_csv(self.data_path / 'training_transactions.csv', index=False)
        
        return merchants_df, transactions_df
    
    def generate_test_data(self, merchant_count: int = 100, 
                          fraud_percentage: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate test data with both normal and fraudulent behavior"""
        logger.info(f"Generating test dataset with {fraud_percentage*100}% fraudulent merchants...")
        
        # Calculate counts
        normal_count = int(merchant_count * (1 - fraud_percentage))
        fraud_count = merchant_count - normal_count
        
        # Generate normal merchants
        normal_merchants, normal_txns = self.generate_training_data(normal_count)
        normal_merchants['is_fraud'] = False
        
        # Generate fraud merchants with patterns
        fraud_merchants = []
        fraud_txns = []
        
        for _ in range(fraud_count):
            merchant = {
                'merchant_id': f"M{uuid.uuid4().hex[:8].upper()}",
                'business_type': random.choice(self.business_types),
                'registration_date': self._generate_registration_date(),
                'gst_status': random.choice(self.gst_statuses),
                'volume_category': random.choice(self.volume_categories),
                'is_fraud': True,
                'fraud_pattern': random.choice([
                    'late_night', 'velocity', 'customer_concentration',
                    'split_transaction', 'round_amount'
                ])
            }
            fraud_merchants.append(merchant)
            
            # Generate fraudulent transactions based on pattern
            merchant_txns = self._generate_fraud_transactions(
                merchant['merchant_id'],
                merchant['fraud_pattern']
            )
            fraud_txns.extend(merchant_txns)
        
        # Combine and shuffle data
        merchants_df = pd.concat([
            normal_merchants,
            pd.DataFrame(fraud_merchants)
        ]).sample(frac=1).reset_index(drop=True)
        
        transactions_df = pd.concat([
            pd.DataFrame(normal_txns),
            pd.DataFrame(fraud_txns)
        ]).sample(frac=1).reset_index(drop=True)
        
        # Save test data
        merchants_df.to_csv(self.data_path / 'test_merchants.csv', index=False)
        transactions_df.to_csv(self.data_path / 'test_transactions.csv', index=False)
        
        return merchants_df, transactions_df

    def _generate_normal_transactions(self, merchant_id: str, 
                                   volume_category: str,
                                   days: int = 30) -> List[Dict]:
        """Generate normal transaction patterns"""
        transactions = []
        current_date = datetime.now() - timedelta(days=days)
        
        # Volume configurations
        volume_configs = {
            'low': (5, 15),
            'medium': (15, 30),
            'high': (30, 50)
        }
        
        daily_range = volume_configs[volume_category]
        
        for _ in range(days):
            num_transactions = random.randint(*daily_range)
            
            for _ in range(num_transactions):
                # Generate during business hours (8 AM - 8 PM)
                hour = random.randint(8, 20)
                timestamp = current_date.replace(
                    hour=hour,
                    minute=random.randint(0, 59)
                )
                
                txn = {
                    'transaction_id': f"T{uuid.uuid4().hex[:12].upper()}",
                    'merchant_id': merchant_id,
                    'timestamp': timestamp,
                    'amount': random.uniform(100, 5000),
                    'customer_id': f"C{random.randint(10000, 99999)}",
                    'status': 'completed'
                }
                transactions.append(txn)
            
            current_date += timedelta(days=1)
        
        return transactions 

    def _generate_registration_date(self) -> datetime:
        """Generate a random registration date for a merchant"""
        days_range = (self.end_date - self.start_date).days
        random_days = random.randint(0, days_range)
        return self.start_date + timedelta(days=random_days)

    def _generate_fraud_transactions(self, merchant_id: str, fraud_pattern: str) -> List[Dict]:
        """Generate fraudulent transactions based on the specified pattern"""
        transactions = []
        current_date = datetime.now() - timedelta(days=30)
        
        if fraud_pattern == 'late_night':
            transactions = self._generate_late_night_pattern(merchant_id, current_date)
        elif fraud_pattern == 'velocity':
            transactions = self._generate_velocity_pattern(merchant_id, current_date)
        elif fraud_pattern == 'customer_concentration':
            transactions = self._generate_customer_concentration_pattern(merchant_id, current_date)
        elif fraud_pattern == 'split_transaction':
            transactions = self._generate_split_transaction_pattern(merchant_id, current_date)
        elif fraud_pattern == 'round_amount':
            transactions = self._generate_round_amount_pattern(merchant_id, current_date)
        
        return transactions

    def _generate_late_night_pattern(self, merchant_id: str, start_date: datetime) -> List[Dict]:
        """Generate transactions with late night pattern"""
        transactions = []
        for _ in range(30):  # 30 days
            num_transactions = random.randint(10, 20)
            for _ in range(num_transactions):
                hour = random.choice([22, 23, 0, 1, 2, 3])
                timestamp = start_date.replace(hour=hour, minute=random.randint(0, 59))
                transactions.append({
                    'transaction_id': f"T{uuid.uuid4().hex[:12].upper()}",
                    'merchant_id': merchant_id,
                    'timestamp': timestamp,
                    'amount': random.uniform(100, 5000),
                    'customer_id': f"C{random.randint(10000, 99999)}",
                    'status': 'completed'
                })
            start_date += timedelta(days=1)
        return transactions

    def _generate_velocity_pattern(self, merchant_id: str, start_date: datetime) -> List[Dict]:
        """Generate transactions with sudden velocity spike"""
        transactions = []
        for day in range(30):
            if day in [10, 11, 12]:  # Spike days
                num_transactions = random.randint(200, 300)
            else:
                num_transactions = random.randint(5, 15)
            
            for _ in range(num_transactions):
                timestamp = start_date.replace(hour=random.randint(8, 20), minute=random.randint(0, 59))
                transactions.append({
                    'transaction_id': f"T{uuid.uuid4().hex[:12].upper()}",
                    'merchant_id': merchant_id,
                    'timestamp': timestamp,
                    'amount': random.uniform(100, 5000),
                    'customer_id': f"C{random.randint(10000, 99999)}",
                    'status': 'completed'
                })
            start_date += timedelta(days=1)
        return transactions

    def _generate_customer_concentration_pattern(self, merchant_id: str, start_date: datetime) -> List[Dict]:
        """Generate transactions with high customer concentration"""
        transactions = []
        main_customers = [f"C{random.randint(10000, 99999)}" for _ in range(3)]
        for _ in range(30):  # 30 days
            num_transactions = random.randint(15, 25)
            for _ in range(num_transactions):
                if random.random() < 0.9:  # 90% transactions from main customers
                    customer_id = random.choice(main_customers)
                else:
                    customer_id = f"C{random.randint(10000, 99999)}"
                timestamp = start_date.replace(hour=random.randint(8, 20), minute=random.randint(0, 59))
                transactions.append({
                    'transaction_id': f"T{uuid.uuid4().hex[:12].upper()}",
                    'merchant_id': merchant_id,
                    'timestamp': timestamp,
                    'amount': random.uniform(100, 5000),
                    'customer_id': customer_id,
                    'status': 'completed'
                })
            start_date += timedelta(days=1)
        return transactions

    def _generate_split_transaction_pattern(self, merchant_id: str, start_date: datetime) -> List[Dict]:
        """Generate transactions with split pattern"""
        transactions = []
        for _ in range(30):  # 30 days
            if random.random() < 0.3:  # 30% chance of split transaction set
                original_amount = random.uniform(10000, 50000)
                num_splits = random.randint(5, 10)
                split_amount = original_amount / num_splits
                customer_id = f"C{random.randint(10000, 99999)}"
                base_minute = random.randint(0, 30)
                hour = random.randint(8, 20)
                
                for i in range(num_splits):
                    timestamp = start_date.replace(hour=hour, minute=(base_minute + i) % 60)
                    transactions.append({
                        'transaction_id': f"T{uuid.uuid4().hex[:12].upper()}",
                        'merchant_id': merchant_id,
                        'timestamp': timestamp,
                        'amount': split_amount,
                        'customer_id': customer_id,
                        'status': 'completed'
                    })
            
            # Add some normal transactions
            num_normal = random.randint(5, 15)
            for _ in range(num_normal):
                timestamp = start_date.replace(hour=random.randint(8, 20), minute=random.randint(0, 59))
                transactions.append({
                    'transaction_id': f"T{uuid.uuid4().hex[:12].upper()}",
                    'merchant_id': merchant_id,
                    'timestamp': timestamp,
                    'amount': random.uniform(100, 5000),
                    'customer_id': f"C{random.randint(10000, 99999)}",
                    'status': 'completed'
                })
            start_date += timedelta(days=1)
        return transactions

    def _generate_round_amount_pattern(self, merchant_id: str, start_date: datetime) -> List[Dict]:
        """Generate transactions with round amount pattern"""
        transactions = []
        round_amounts = [999, 1999, 4999, 9999, 19999]
        for _ in range(30):  # 30 days
            num_transactions = random.randint(10, 20)
            for _ in range(num_transactions):
                if random.random() < 0.7:  # 70% chance of round amount
                    amount = random.choice(round_amounts)
                else:
                    amount = random.uniform(100, 5000)
                timestamp = start_date.replace(hour=random.randint(8, 20), minute=random.randint(0, 59))
                transactions.append({
                    'transaction_id': f"T{uuid.uuid4().hex[:12].upper()}",
                    'merchant_id': merchant_id,
                    'timestamp': timestamp,
                    'amount': amount,
                    'customer_id': f"C{random.randint(10000, 99999)}",
                    'status': 'completed'
                })
            start_date += timedelta(days=1)
        return transactions