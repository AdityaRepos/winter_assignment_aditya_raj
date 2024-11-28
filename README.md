# Merchant Fraud Detection System

## Quick Start
``` bash
cd src
python main.py
```
## Core Components
1. **Data Generation** (`data_generation.py`)
   - Creates synthetic merchant profiles
   - Generates transaction patterns
   - Injects fraud scenarios

2. **Main Pipeline** (`main.py`)
   - Orchestrates fraud detection process
   - Manages data and model execution

3. **Feature Engineering** (`feature_engineering.py`)
   - Processes transactions
   - Extracts merchant patterns

4. **Fraud Detection** (`fraud_detection.py`, `anomaly_detection.py`)
   - Autoencoder model implementation
   - Pattern matching and risk scoring

5. **Pattern Analysis** (`fraud_patterns.py`)
   - Fraud indicators and validation

## Key Features

### 1. Data Simulation
- Merchant profile generation
- Normal transaction patterns
- Fraudulent behavior injection

### 2. Transaction Monitoring
- Transaction velocity and volume analysis
- Amount pattern detection
- Time-based pattern analysis

### 3. Merchant Analysis
- Business pattern monitoring
- Customer behavior tracking
- Geographic distribution analysis

### 4. Anomaly Detection
- Statistical outlier detection
- Autoencoder-based pattern learning
- Risk score calculation

## Outputs (`/data`)
- **Visualizations**: Fraud distribution and model metrics
- **Reports**: Detailed analysis and executive summary

## Current Status & Future Plans
### Current Implementation
- Synthetic data generation and basic autoencoder
- Pattern matching and reporting system

### Key Improvements Planned
1. Advanced ML models (Transformers, LSTM)
2. Real transaction data integration
3. Interactive dashboards and API endpoints
4. Real-time processing capabilities
5. Production-ready monitoring system
