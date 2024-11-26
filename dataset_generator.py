import random
from typing import List, Tuple
from datetime import datetime, timedelta
import string

# Constants
BUSINESS_TYPES = [
    "Retail", "Restaurant", "E-commerce", "Services", 
    "Manufacturing", "Technology", "Healthcare", "Education"
]

PRODUCT_CATEGORIES = [
    "Electronics", "Fashion", "Food & Beverages", "Home & Living",
    "Health & Beauty", "Sports & Outdoors", "Books & Media", "Automotive"
]

CITIES = [
    "Mumbai", "Delhi", "Bangalore", "Chennai", "Kolkata", 
    "Hyderabad", "Pune", "Ahmedabad", "Jaipur", "Lucknow"
]

STATES = [
    "Maharashtra", "Delhi", "Karnataka", "Tamil Nadu", "West Bengal",
    "Telangana", "Gujarat", "Rajasthan", "Uttar Pradesh", "Kerala"
]

PAYMENT_METHODS = [
    "Credit Card", "Debit Card", "UPI", "Net Banking",
    "Wallet", "Cash", "EMI", "BNPL"
]

# Helper functions
def generate_merchant_id() -> str:
    return f"M{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"

def generate_business_name() -> str:
    prefixes = ["Tech", "Star", "Global", "Indian", "Smart", "Prime", "Royal", "Super"]
    suffixes = ["Solutions", "Enterprises", "Trading", "Services", "Industries", "Retail"]
    return f"{random.choice(prefixes)} {random.choice(suffixes)}"

def generate_random_date() -> str:
    start_date = datetime(2015, 1, 1)
    end_date = datetime.now()
    time_between = end_date - start_date
    random_days = random.randint(0, time_between.days)
    return (start_date + timedelta(days=random_days)).strftime("%Y-%m-%d")

def generate_pan_number() -> str:
    return f"{''.join(random.choices(string.ascii_uppercase, k=5))}{''.join(random.choices(string.digits, k=4))}{''.join(random.choices(string.ascii_uppercase, k=1))}"

def generate_address() -> str:
    street_numbers = random.randint(1, 999)
    streets = ["Main Road", "Park Street", "MG Road", "Ring Road", "Commercial Street"]
    areas = ["Andheri", "Bandra", "Malad", "Powai", "Worli"]
    return f"{street_numbers}, {random.choice(streets)}, {random.choice(areas)}"

def generate_bank_account() -> str:
    account_number = ''.join(random.choices(string.digits, k=12))
    ifsc_code = f"{''.join(random.choices(string.ascii_uppercase, k=4))}{''.join(random.choices(string.digits, k=6))}"
    bank_name = random.choice(["HDFC", "ICICI", "SBI", "Axis", "Kotak"])
    return f"{account_number}|{ifsc_code}|{bank_name}"

def generate_txn_id() -> str:
    return f"T{''.join(random.choices(string.ascii_uppercase + string.digits, k=10))}"

def generate_customer_id() -> str:
    return f"C{''.join(random.choices(string.ascii_uppercase + string.digits, k=8))}"

def generate_device_id() -> str:
    return f"D{''.join(random.choices(string.ascii_uppercase + string.digits, k=10))}"

def generate_business_hour_timestamp() -> str:
    current_date = datetime.now().date()
    random_hour = random.randint(9, 21)  # Business hours between 9 AM and 9 PM
    random_minute = random.randint(0, 59)
    random_second = random.randint(0, 59)
    return datetime.combine(current_date, datetime.min.time().replace(
        hour=random_hour, minute=random_minute, second=random_second
    )).strftime("%Y-%m-%d %H:%M:%S")

def generate_merchant_base(count: int) -> List[dict]:
    """Generate base merchant profiles"""
    merchants = []
    for _ in range(count):
        merchant = {
            "merchant_id": generate_merchant_id(),
            "business_name": generate_business_name(),
            "business_type": random.choice(BUSINESS_TYPES),
            "registration_date": generate_random_date(),
            "business_model": random.choice(["Online", "Offline", "Hybrid"]),
            "product_category": random.choice(PRODUCT_CATEGORIES),
            "average_ticket_size": random.uniform(100, 5000),
            "gst_status": random.choice([True, False]),
            "pan_number": generate_pan_number(),
            "epfo_registered": random.choice([True, False]),
            "registered_address": generate_address(),
            "city": random.choice(CITIES),
            "state": random.choice(STATES),
            "reported_revenue": random.uniform(100000, 10000000),
            "employee_count": random.randint(1, 100),
            "bank_account": generate_bank_account()
        }
        merchants.append(merchant)
    return merchants

def generate_normal_transactions(
    merchant_id: str,
    days: int,
    daily_volume: Tuple[int, int],
    amount_range: Tuple[float, float]
) -> List[dict]:
    """Generate normal transaction patterns"""
    transactions = []
    for _ in range(days):
        daily_txns = random.randint(*daily_volume)
        for _ in range(daily_txns):
            txn = {
                "transaction_id": generate_txn_id(),
                "merchant_id": merchant_id,
                "amount": random.uniform(*amount_range),
                "timestamp": generate_business_hour_timestamp(),
                "customer_id": generate_customer_id(),
                "device_id": generate_device_id(),
                "customer_location": random.choice(CITIES),
                "payment_method": random.choice(PAYMENT_METHODS),
                "status": random.choice(["completed", "failed", "refunded"]),
                "product_category": random.choice(PRODUCT_CATEGORIES),
                "platform": random.choice(["Web", "Mobile", "POS"]),
                "velocity_flag": False,
                "amount_flag": False,
                "time_flag": False,
                "device_flag": False
            }
            transactions.append(txn)
    return transactions

def inject_late_night_pattern(transactions: List[dict], config: dict) -> List[dict]:
    """Inject late-night trading pattern into transactions"""
    # Calculate pattern duration in days (2-3 weeks)
    pattern_days = random.randint(14, 21)
    
    # Group transactions by date
    txn_by_date = {}
    for txn in transactions:
        date = txn['timestamp'].split()[0]
        txn_by_date.setdefault(date, []).append(txn)
    
    # Select random consecutive days for pattern
    dates = sorted(txn_by_date.keys())
    # Add this check to prevent negative range
    if len(dates) <= pattern_days:
        pattern_days = len(dates) // 2  # Use half the available dates if not enough
    
    pattern_start = random.randint(0, len(dates) - pattern_days)
    pattern_dates = dates[pattern_start:pattern_start + pattern_days]
    
    for date in pattern_dates:
        if len(txn_by_date[date]) < config['characteristics']['min_daily_transactions']:
            continue
            
        # Convert percentage of daily transactions to late night
        num_txns = int(len(txn_by_date[date]) * config['characteristics']['volume_percentage'] / 100)
        selected_txns = random.sample(txn_by_date[date], num_txns)
        
        for txn in selected_txns:
            # Generate late night timestamp
            current_date = datetime.strptime(txn['timestamp'], "%Y-%m-%d %H:%M:%S").date()
            hour = random.randint(23, 28) % 24
            minute = random.randint(0, 59)
            
            new_timestamp = datetime.combine(current_date, datetime.min.time().replace(
                hour=hour, minute=minute
            ))
            if hour < 23:
                new_timestamp += timedelta(days=1)
                
            txn['timestamp'] = new_timestamp.strftime("%Y-%m-%d %H:%M:%S")
            txn['time_flag'] = True
            txn['amount'] *= random.uniform(1.5, 2.0)
            txn['amount_flag'] = True
    
    return transactions

def inject_spike_pattern(transactions: List[dict], config: dict) -> List[dict]:
    """Inject sudden activity spike pattern into transactions"""
    normal_range = [int(x) for x in config['characteristics']['normal_daily_txns'].split('-')]
    spike_range = [int(x) for x in config['characteristics']['spike_daily_txns'].split('-')]
    spike_duration = random.randint(2, 3)  # 2-3 days
    
    # Group transactions by date
    txn_by_date = {}
    for txn in transactions:
        date = txn['timestamp'].split()[0]
        txn_by_date.setdefault(date, []).append(txn)
    
    dates = sorted(txn_by_date.keys())
    # Create spikes every 2-3 weeks
    for i in range(0, len(dates), random.randint(14, 21)):
        if i + spike_duration >= len(dates):
            break
            
        # Increase transaction volume for spike duration
        for j in range(spike_duration):
            date = dates[i + j]
            current_txns = txn_by_date[date]
            
            # Add additional transactions to reach spike volume
            spike_volume = random.randint(*spike_range)
            while len(current_txns) < spike_volume:
                new_txn = current_txns[0].copy()
                new_txn.update({
                    'transaction_id': generate_txn_id(),
                    'timestamp': generate_business_hour_timestamp(),
                    'velocity_flag': True
                })
                current_txns.append(new_txn)
                
    return [txn for txns in txn_by_date.values() for txn in txns]

def inject_split_transactions(transactions: List[dict], config: dict) -> List[dict]:
    """Inject split transaction pattern into transactions"""
    amount_range = [int(x) for x in config['characteristics']['original_amount'].split('-')]
    split_range = [int(x) for x in config['characteristics']['split_count'].split('-')]
    time_window = [int(x) for x in config['characteristics']['time_window'].split('-')]
    
    # Select random transactions to split
    num_splits = len(transactions) // 20  # 5% of transactions will be split
    split_base_txns = random.sample(transactions, num_splits)
    
    new_transactions = []
    for txn in transactions:
        if txn in split_base_txns:
            # Generate split transactions
            original_amount = random.randint(*amount_range)
            num_splits = random.randint(*split_range)
            split_amount = original_amount / num_splits
            
            base_timestamp = datetime.strptime(txn['timestamp'], "%Y-%m-%d %H:%M:%S")
            customer_id = generate_customer_id()
            device_id = generate_device_id()
            
            for i in range(num_splits):
                split_txn = txn.copy()
                split_txn.update({
                    'transaction_id': generate_txn_id(),
                    'amount': split_amount,
                    'timestamp': (base_timestamp + timedelta(
                        minutes=random.randint(0, time_window[1])
                    )).strftime("%Y-%m-%d %H:%M:%S"),
                    'customer_id': customer_id,
                    'device_id': device_id,
                    'amount_flag': True,
                    'velocity_flag': True
                })
                new_transactions.append(split_txn)
        else:
            new_transactions.append(txn)
    
    return new_transactions

def inject_round_amount_pattern(transactions: List[dict], config: dict) -> List[dict]:
    """Inject round amount pattern into transactions"""
    amount_patterns = config['characteristics']['amount_pattern']
    frequency = int(config['characteristics']['frequency'].split('%')[0])
    
    num_txns = int(len(transactions) * frequency / 100)
    selected_txns = random.sample(transactions, num_txns)
    
    for txn in transactions:
        if txn in selected_txns:
            txn['amount'] = random.choice(amount_patterns)
            txn['amount_flag'] = True
    
    return transactions

def inject_customer_concentration(transactions: List[dict], config: dict) -> List[dict]:
    """Inject customer concentration pattern into transactions"""
    num_customers = random.randint(5, 10)
    volume_concentration = int(config['characteristics']['volume_concentration'].split('%')[0])
    
    # Generate suspicious customer profiles
    suspicious_customers = []
    for _ in range(num_customers):
        suspicious_customers.append({
            'customer_id': generate_customer_id(),
            'device_ids': [generate_device_id() for _ in range(2)],
            'location': random.choice(CITIES)
        })
    
    # Select transactions for concentration
    num_txns = int(len(transactions) * volume_concentration / 100)
    selected_txns = random.sample(transactions, num_txns)
    
    for txn in transactions:
        if txn in selected_txns:
            customer = random.choice(suspicious_customers)
            txn.update({
                'customer_id': customer['customer_id'],
                'device_id': random.choice(customer['device_ids']),
                'customer_location': customer['location'],
                'device_flag': True,
                'velocity_flag': True
            })
    
    return transactions

def inject_fraud_pattern(
    transactions: List[dict],
    pattern_type: str,
    config: dict
) -> List[dict]:
    """Inject specific fraud pattern into transaction list"""
    pattern_functions = {
        "late_night_trading": inject_late_night_pattern,
        "sudden_activity_spike": inject_spike_pattern,
        "split_transactions": inject_split_transactions,
        "round_amount_pattern": inject_round_amount_pattern,
        "customer_concentration": inject_customer_concentration
    }
    
    if pattern_type in pattern_functions:
        return pattern_functions[pattern_type](transactions, config)
    return transactions

def generate_fraudulent_transactions(merchant: dict, pattern: str) -> List[dict]:
    """Generate fraudulent transactions for a merchant based on pattern"""
    pattern_configs = {
        "late_night_trading": {
            "characteristics": {
                "time_window": "23:00-04:00",
                "volume_percentage": 90,
                "min_daily_transactions": 50,
                "pattern_duration": "2-3 weeks"
            }
        },
        "sudden_activity_spike": {
            "characteristics": {
                "normal_daily_txns": "10-20",
                "spike_daily_txns": "500-1000",
                "spike_duration": "2-3 days",
                "pattern_frequency": "Once every 2-3 weeks"
            }
        },
        "split_transactions": {
            "characteristics": {
                "original_amount": "100000-500000",
                "split_count": "20-30",
                "time_window": "5-15",
                "same_customer": True
            }
        },
        "round_amount_pattern": {
            "characteristics": {
                "amount_pattern": [99999, 199999, 299999],
                "frequency": "90%",
                "time_window": "All day"
            }
        },
        "customer_concentration": {
            "characteristics": {
                "customer_count": "2-3",
                "volume_concentration": "95%",
                "regular_frequency": "Daily transactions",
                "relationship": "Common device IDs/locations"
            }
        }
    }
    
    # Generate fewer base transactions for fraud patterns
    base_transactions = generate_normal_transactions(
        merchant_id=merchant['merchant_id'],
        days=30,
        daily_volume=(5, 15),
        amount_range=(1000, 10000)
    )
    
    return inject_fraud_pattern(base_transactions, pattern, pattern_configs[pattern])

def generate_dataset(
    merchant_count: int,
    fraud_percentage: float,
    patterns: List[str]
) -> Tuple[List[dict], List[dict]]:
    """Generate complete dataset with mix of normal and fraudulent patterns"""
    merchants = generate_merchant_base(merchant_count)
    
    fraud_count = int(merchant_count * fraud_percentage)
    fraud_merchants = random.sample(merchants, fraud_count)
    
    all_transactions = []
    for merchant in merchants:
        if merchant in fraud_merchants:
            pattern = random.choice(patterns)
            txns = generate_fraudulent_transactions(merchant, pattern)
        else:
            txns = generate_normal_transactions(
                merchant_id=merchant['merchant_id'],
                days=30,
                daily_volume=(10, 50),
                amount_range=(100, 5000)
            )
        all_transactions.extend(txns)
    
    return merchants, all_transactions

import pandas as pd

def save_to_csv(merchants, transactions, output_dir="./"):
    """
    Save merchants and transactions data to CSV files
    
    Args:
        merchants (List[dict]): List of merchant dictionaries
        transactions (List[dict]): List of transaction dictionaries
        output_dir (str): Directory to save the CSV files (defaults to current directory)
    """
    # Convert to pandas DataFrames
    merchants_df = pd.DataFrame(merchants)
    transactions_df = pd.DataFrame(transactions)
    
    # Save to CSV
    merchants_df.to_csv(f"{output_dir}merchants.csv", index=False)
    transactions_df.to_csv(f"{output_dir}transactions.csv", index=False)

# Generate and save the data
if __name__ == "__main__":
    merchants, transactions = generate_dataset(
        merchant_count=100,
        fraud_percentage=0.1,
        patterns=[
            "late_night_trading",
            "sudden_activity_spike",
            "split_transactions",
            "round_amount_pattern",
            "customer_concentration"
        ]
    )
    
    save_to_csv(merchants, transactions)