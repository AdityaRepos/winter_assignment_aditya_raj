# Merchant Fraud Detection System - Code Overview

## Basic usage with default settings
`python src/main.py`

## Core Components & Flow

### 1. Main Pipeline (`src/main.py`)
The `FraudDetectionPipeline` class orchestrates the entire process:
- Initializes data paths and components
- Manages data generation and processing
- Handles model training and prediction
- Generates fraud reports

### 2. Feature Processing (`src/feature_engineering.py`)
The `FeatureEngineering` class:
- Processes raw transaction data
- Calculates merchant behavior metrics
- Normalizes features for model input

### 3. Data Flow
Raw Data → Feature Engineering → Model Training → Fraud Detection → Reports


### 4. Key Operations
1. **Data Generation**
   - Creates merchant profiles
   - Generates transaction data
   - Injects fraud patterns

2. **Feature Creation**
   - Transaction metrics
   - Time patterns
   - Customer behavior

3. **Fraud Detection**
   - Pattern matching
   - Anomaly scoring
   - Classification

4. **Reporting**
   - Detailed merchant analysis
   - Fraud pattern summary
   - Visual analytics

### 5. Output Generation
- Generates two main files in `/data`:
  - Detailed fraud report
  - Fraud merchant summary
- Creates visualization files:
  - Fraud pattern distribution chart
  - Merchant risk assessment plot

This system provides an end-to-end solution for detecting fraudulent merchant behavior through pattern analysis and anomaly detection.