# Merchant Fraud Detection System

## Overview
An autoencoder-based anomaly detection system that identifies suspicious merchant behavior patterns by analyzing transaction patterns and characteristics.

## Key Components

### 1. Data Generation
- **Merchant Profiles**: Basic info (ID, name, type) + business details
- **Transaction Data**: 
  - 80% normal trading patterns
  - 20% fraudulent patterns
- **Data Location**: All data stored in external `/data` directory

### 2. Fraud Pattern Detection
Identifies five main fraud patterns:

1. **Late Night Trading**
   - High volume during 23:00-04:00
   - 70%+ transactions in off-hours

2. **Sudden Activity Spikes**
   - Normal: 10-20 daily transactions
   - Spike: 200-300 daily transactions
   - Duration: 2-3 days

3. **Split Transactions**
   - Original amount: 50,000-100,000
   - Split into 5-10 smaller transactions
   - Within 10-30 minutes window

4. **Round Amount Pattern**
   - Just-below threshold amounts (9999, 19999)
   - High frequency of specific amounts

5. **Customer Concentration**
   - 5-10 customers making 80% of volume
   - Regular high-value transactions

### 3. Feature Engineering
Key metrics calculated:
- Transaction velocity
- Time-based patterns
- Amount distributions
- Customer concentration

### 4. Output Files
Located in `/data` directory:

1. **fraud_merchant_report.txt**
   - Detailed analysis per merchant
   - Transaction patterns
   - Customer analysis
   - Fraud type classification

2. **fraud_merchant_summary.txt**
   - Concise list of fraud merchants
   - Key metrics and fraud types

## Testing Scenarios

### 1. Normal Merchant Validation
- Business hour patterns
- Amount distributions
- Customer diversity

### 2. Fraud Pattern Validation
- Pattern characteristics
- Timing analysis
- Pattern intensity

### 3. Dataset Balance
- Fraud/normal ratio
- Pattern distribution
- Overall statistics

## Project Structure

src/
├── main.py # Main pipeline
├── feature_engineering.py # Feature processing
└── data/ # Output directory


## Future Improvements
1. **Pattern Detection**
   - Additional fraud patterns
   - Enhanced detection algorithms

2. **Feature Engineering**
   - More temporal features
   - Advanced customer metrics

3. **Reporting**
   - Visualization capabilities
   - Real-time monitoring
