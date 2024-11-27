from datetime import datetime, timedelta
import random
from typing import List, Dict
import numpy as np

class FraudPatternInjector:
    def __init__(self, seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        
        # Enhanced pattern configurations
        self.pattern_configs = {
            'late_night': {
                'night_hours': [23, 0, 1, 2, 3, 4],
                'night_ratio': 0.8,  # 80% transactions during night
                'min_transactions': 30,
                'amount_range': (5000, 15000),  # Higher amounts during night
                'time_window_days': 14
            },
            'sudden_spike': {
                'spike_days': 3,  # Concentrated spike period
                'normal_daily_range': (5, 15),
                'spike_daily_range': (200, 300),  # Much higher volume
                'amount_range': (1000, 5000)
            },
            'split_transaction': {
                'original_amount_range': (50000, 100000),
                'split_count_range': (8, 12),
                'time_window_minutes': 30,
                'num_occurrences': 15  # Number of split transaction sets
            },
            'round_amount': {
                'amounts': [9999, 19999, 29999, 49999, 99999],
                'round_ratio': 0.9,  # 90% of transactions should be round
                'daily_transactions': (10, 20),
                'normal_amount_range': (1000, 5000)
            },
            'customer_concentration': {
                'num_customers': 3,  # Very few unique customers
                'concentration_ratio': 0.9,  # 90% from same customers
                'daily_transactions': (15, 25),
                'amount_range': (5000, 15000)
            }
        }

    def _generate_late_night_timestamp(self, date: datetime) -> datetime:
        """Generate timestamp during late night hours (23:00-04:00)"""
        try:
            if random.random() < 0.5:
                hour = 23
            else:
                hour = random.randint(0, 4)
            
            minute = random.randint(0, 59)
            return date.replace(hour=hour, minute=minute)
        except ValueError as e:
            print(f"Error generating timestamp: {e}")
            # Return a safe default value
            return date.replace(hour=23, minute=0)

    def _generate_transaction_id(self) -> str:
        """Generate unique transaction ID"""
        return f"T{random.randint(100000, 999999)}"

    def inject_late_night_pattern(self, transactions: List[Dict], 
                                merchant_id: str, 
                                start_date: datetime) -> List[Dict]:
        """Enhanced late night pattern injection"""
        pattern_transactions = []
        config = self.pattern_configs['late_night']
        current_date = start_date

        for _ in range(config['time_window_days']):
            num_transactions = random.randint(20, 30)
            night_transactions = int(num_transactions * config['night_ratio'])
            
            # Generate night transactions
            for _ in range(night_transactions):
                hour = random.choice(config['night_hours'])
                timestamp = current_date.replace(
                    hour=hour,
                    minute=random.randint(0, 59)
                )
                
                txn = {
                    "transaction_id": self._generate_transaction_id(),
                    "merchant_id": merchant_id,
                    "timestamp": timestamp,
                    "amount": random.uniform(*config['amount_range']),
                    "customer_id": f"C{random.randint(10000, 99999)}",
                    "device_id": f"D{random.randint(1000, 9999)}",
                    "status": "completed"
                }
                pattern_transactions.append(txn)
            
            # Generate some day transactions
            for _ in range(num_transactions - night_transactions):
                hour = random.randint(9, 22)
                timestamp = current_date.replace(
                    hour=hour,
                    minute=random.randint(0, 59)
                )
                
                txn = {
                    "transaction_id": self._generate_transaction_id(),
                    "merchant_id": merchant_id,
                    "timestamp": timestamp,
                    "amount": random.uniform(1000, 5000),
                    "customer_id": f"C{random.randint(10000, 99999)}",
                    "device_id": f"D{random.randint(1000, 9999)}",
                    "status": "completed"
                }
                pattern_transactions.append(txn)
            
            current_date += timedelta(days=1)

        return transactions + pattern_transactions

    def inject_sudden_spike_pattern(self, transactions: List[Dict], 
                                  merchant_id: str,
                                  start_date: datetime,
                                  spike_duration: int = 3) -> List[Dict]:
        """Inject sudden activity spike pattern"""
        pattern_transactions = []
        current_date = start_date
        config = self.pattern_configs['sudden_spike']

        for _ in range(spike_duration):
            num_transactions = random.randint(
                config['min_transactions'],
                config['max_transactions']
            )
            
            for _ in range(num_transactions):
                hour = random.randint(9, 21)
                timestamp = current_date.replace(
                    hour=hour,
                    minute=random.randint(0, 59)
                )
                txn = {
                    "transaction_id": self._generate_transaction_id(),
                    "merchant_id": merchant_id,
                    "timestamp": timestamp,
                    "amount": random.uniform(*config['amount_range']),
                    "customer_id": f"C{random.randint(10000, 99999)}",
                    "device_id": f"D{random.randint(1000, 9999)}",
                    "status": "completed",
                    "time_flag": False,
                    "velocity_flag": True,
                    "amount_flag": False,
                    "device_flag": False
                }
                pattern_transactions.append(txn)
            
            current_date += timedelta(days=1)

        return transactions + pattern_transactions

    def inject_split_transactions(self, transactions: List[Dict], 
                                merchant_id: str,
                                start_date: datetime) -> List[Dict]:
        """Enhanced split transaction pattern injection"""
        pattern_transactions = []
        config = self.pattern_configs['split_transaction']
        current_date = start_date

        for _ in range(config['num_occurrences']):
            # Generate original large amount
            original_amount = random.uniform(*config['original_amount_range'])
            num_splits = random.randint(*config['split_count_range'])
            split_amount = original_amount / num_splits
            
            # Generate split transactions
            base_minute = random.randint(0, 59)
            hour = random.randint(9, 20)
            customer_id = f"C{random.randint(10000, 99999)}"
            device_id = f"D{random.randint(1000, 9999)}"
            
            for i in range(num_splits):
                timestamp = current_date.replace(
                    hour=hour,
                    minute=(base_minute + i) % 60
                )
                
                txn = {
                    "transaction_id": self._generate_transaction_id(),
                    "merchant_id": merchant_id,
                    "timestamp": timestamp,
                    "amount": split_amount,
                    "customer_id": customer_id,
                    "device_id": device_id,
                    "status": "completed"
                }
                pattern_transactions.append(txn)
            
            current_date += timedelta(days=random.randint(1, 3))

        return transactions + pattern_transactions

    def inject_round_amount_pattern(self, transactions: List[Dict], 
                                  merchant_id: str,
                                  start_date: datetime,
                                  duration_days: int = 7) -> List[Dict]:
        """Inject round amount pattern"""
        pattern_transactions = []
        current_date = start_date
        config = self.pattern_configs['round_amount']

        for _ in range(duration_days):
            num_transactions = random.randint(*config['daily_transactions'])
            
            for _ in range(num_transactions):
                hour = random.randint(9, 21)
                timestamp = current_date.replace(
                    hour=hour,
                    minute=random.randint(0, 59)
                )
                txn = {
                    "transaction_id": self._generate_transaction_id(),
                    "merchant_id": merchant_id,
                    "timestamp": timestamp,
                    "amount": random.choice(config['amounts']),
                    "customer_id": f"C{random.randint(10000, 99999)}",
                    "device_id": f"D{random.randint(1000, 9999)}",
                    "status": "completed",
                    "time_flag": False,
                    "velocity_flag": False,
                    "amount_flag": True,
                    "device_flag": False
                }
                pattern_transactions.append(txn)
            
            current_date += timedelta(days=1)

        return transactions + pattern_transactions

    def inject_random_pattern(self, transactions: List[Dict], 
                            merchant_id: str,
                            start_date: datetime) -> List[Dict]:
        """Inject a random fraud pattern with equal probability"""
        patterns = [
            (self.inject_late_night_pattern, 'late_night'),
            (self.inject_sudden_spike_pattern, 'sudden_spike'),
            (self.inject_split_transactions, 'split_transaction'),
            (self.inject_round_amount_pattern, 'round_amount'),
            (self.inject_customer_concentration, 'customer_concentration')
        ]
        
        # Ensure equal distribution of patterns
        chosen_pattern, pattern_name = random.choice(patterns)
        print(f"Injecting {pattern_name} pattern for merchant {merchant_id}")
        
        try:
            return chosen_pattern(transactions, merchant_id, start_date)
        except Exception as e:
            print(f"Error injecting pattern: {e}")
            return transactions

    def inject_customer_concentration(self, transactions: List[Dict], 
                                merchant_id: str,
                                start_date: datetime,
                                duration_days: int = 14) -> List[Dict]:
        """Inject customer concentration pattern"""
        pattern_transactions = []
        config = self.pattern_configs['customer_concentration']
        
        # Generate a small pool of customers and devices
        customers = [f"C{random.randint(10000, 99999)}" for _ in range(config['num_customers'])]
        devices = [f"D{random.randint(1000, 9999)}" for _ in range(3)]
        current_date = start_date

        for _ in range(duration_days):
            # Generate more transactions from the small customer pool
            num_transactions = random.randint(15, 25)
            
            for _ in range(num_transactions):
                hour = random.randint(9, 21)
                timestamp = current_date.replace(
                    hour=hour,
                    minute=random.randint(0, 59)
                )
                
                # Use concentrated customer pool
                customer_id = random.choice(customers)
                
                txn = {
                    "transaction_id": self._generate_transaction_id(),
                    "merchant_id": merchant_id,
                    "timestamp": timestamp,
                    "amount": random.uniform(*config['amount_range']),
                    "customer_id": customer_id,
                    "device_id": random.choice(devices),
                    "status": "completed",
                    "time_flag": False,
                    "velocity_flag": False,
                    "amount_flag": False,
                    "device_flag": True
                }
                pattern_transactions.append(txn)
            
            current_date += timedelta(days=1)

        return transactions + pattern_transactions