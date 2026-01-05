# 🛡️ Real-Time Fraud Detection System

An end-to-end machine learning system for detecting fraudulent credit card transactions in real-time, achieving 90% fraud detection with less than 1% false alarm rate.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

This project demonstrates a production-ready fraud detection system built from scratch, covering the complete ML lifecycle from data exploration to API deployment. The system processes transactions in real-time, leveraging historical customer and terminal behavior to identify fraudulent patterns.

### Key Features
-  **Real-time predictions** in under 1 second
-  **90.2% fraud recall** with 83.3% precision
-  **Feature store architecture** for low-latency inference
-  **Asynchronous background processing** for feature updates
-  **Comprehensive model evaluation** with proper time-series validation


## 📈 Model Performance

| Model | ROC-AUC | PR-AUC | Recall | Precision | Alert Rate | False Alarms |
|-------|---------|--------|--------|-----------|------------|--------------|
| **XGBoost** | **0.9849** | **0.9237** | **90.2%** | **83.3%** | **0.96%** | **431** |
| Random Forest | 0.9841 | 0.8987 | 90.3% | 64.2% | 1.25% | 1,198 |
| LightGBM | 0.9791 | 0.7508 | 90.9% | 63.6% | 1.27% | 1,240 |
| Logistic Regression | 0.9845 | 0.7452 | 91.1% | 52.2% | 1.55% | 1,991 |

### Business Impact
- **2,149 out of 2,382 frauds caught** (90.2% detection rate)
- **Only 431 false alarms** (83.3% of flagged transactions are actual fraud)
- **0.96% alert rate** (flags less than 1% of transactions for manual review)
- **Operationally viable** for production deployment

---

## 🏗️ System Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                    Transaction Input                         │
│              (customer_id, terminal_id, amount)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feature Extraction                         │
│  • Temporal: Hour, day of week, cyclical encoding           │
│  • Amount: Value, high amount flags                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Store (Supabase/PostgreSQL)             │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │ Customer Features│         │ Terminal Features│         │
│  │ • Avg amount 7d  │         │ • Fraud rate 7d  │         │
│  │ • Tx count 30d   │         │ • Tx count 30d   │         │
│  │ • Std deviation  │         │ • High risk flag │         │
│  └──────────────────┘         └──────────────────┘         │
└────────────────────────┬────────────────────────────────────┘
                         │ 
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Feature Enrichment                         │
│  • Combine real-time + historical features                  │
│  • Calculate deviation scores                                │
│  • Compute risk scores                                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  XGBoost Model Inference                     │
│              (30+ features → fraud probability)              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                      Decision Output                         │
│  fraud_probability: 0.0-1.0                                 │
│  is_fraud: true/false (threshold: 0.9348)                   │
│  risk_level: LOW/MEDIUM/HIGH/CRITICAL                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Background Processing (Async)                     │
│  • Log transaction to database                              │
│  • Queue for feature store update                           │
│  • Update customer/terminal statistics                      │
└─────────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Features

### Data Source
- **Simulated transaction data** from [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/)
- **1,754,155 transactions** over 183 days
- **5,000 customers** and **10,000 terminals**
- **0.84% fraud rate** (14,681 fraudulent transactions)

### Fraud Scenarios
1. **Scenario 1**: High amount transactions (>$220)
2. **Scenario 2**: Compromised terminals (2 terminals/day × 28 days)
3. **Scenario 3**: Compromised cards (3 customers/day × 14 days)

### Feature Engineering (30+ features)

#### Temporal Features
- Hour of day (sine/cosine encoding for cyclical nature)
- Day of week
- Weekend indicator
- Night transaction flag (10pm-6am)

#### Customer Aggregations (1-day, 7-day, 30-day windows)
- Transaction count
- Average transaction amount
- Standard deviation of amounts
- Amount deviation from normal behavior
- High deviation flag (>2 standard deviations)

#### Terminal Aggregations (1-day, 7-day, 30-day windows)
- Transaction count
- Fraud rate (percentage of fraudulent transactions)
- Average transaction amount
- High-risk terminal flag (>5% fraud rate)

#### Interaction Features
- Night + high amount combination
- Combined risk score (weighted sum of risk indicators)

---

## 🔬 Methodology

### 1. Data Leakage Prevention
**Critical insight**: Standard rolling windows cause data leakage in time-series data.
```python
# Wrong: Future data leaks into training
df['avg_amount'] = df['amount'].rolling(window=7).mean()

# Correct Method: Shift by 1 to use only past data
df['avg_amount'] = df['amount'].shift(1).rolling(window=7).mean()
```

### 2. Time-Based Train/Val/Test Split
- **Train**: First 70% of days (chronological)
- **Validation**: Next 15% of days
- **Test**: Last 15% of days

Ensures model is evaluated on future data it hasn't seen, mimicking production conditions.

### 3. Handling Class Imbalance
- **Class weighting**: `scale_pos_weight` for tree models
- **Threshold optimization**: Tuned to balance recall vs precision
- **Evaluation metrics**: PR-AUC (more informative than ROC-AUC for imbalanced data)

### 4. Model Calibration
Applied isotonic calibration to all models to ensure fraud probabilities are well-calibrated.

---


## 🧪 Testing

Run the comprehensive test suite:
```bash
python test_api.py
```

This will:
- Test all API endpoints
- Run 20+ prediction scenarios
- Measure performance consistency
- Generate `test_results.csv` with detailed results

### Example Test Output
```
TEST 1: Health Check ✅
TEST 2: System Statistics ✅
TEST 3: Legitimate Transaction ✅
  Fraud Probability: 0.0467
  Decision: LEGITIMATE
  Processing Time: 42.3ms

TEST 4: Suspicious High Amount 🚨
  Fraud Probability: 0.9988
  Decision: FRAUD
  Processing Time: 38.7ms
```

---

## 🔑 Key Learnings & Challenges

### 1. Data Leakage in Time-Series
**Problem**: Initial model achieved 100% accuracy which cannot be right  
**Root Cause**: Rolling windows computed on sorted data before splitting  
**Solution**: Implemented shifted rolling windows using `.shift(1)`

### 2. Feature Store Design
**Challenge**: Need historical customer/terminal data for real-time predictions  
**Solution**: Pre-compute aggregations and store in database for 10ms lookup

### 3. Threshold Optimization
**Challenge**: Balance between catching fraud (recall) and minimizing false alarms (precision)  
**Solution**: Optimize threshold to achieve business constraints (e.g., <2% alert rate)

### 4. Class Imbalance
**Challenge**: Only 0.84% fraud rate in dataset  
**Solution**: Used PR-AUC metric, class weighting, and proper stratification

---

## Future Improvements

### Performance Optimization
- [ ] Replace PostgreSQL with **Redis** for <10ms feature lookup
- [ ] Implement **connection pooling** to reduce database overhead
- [ ] Add **caching layer** for frequently accessed features

### Production Enhancements
- [ ] Add **Kafka** for real-time streaming instead of REST API
- [ ] Implement **A/B testing** framework for model comparison
- [ ] Create **monitoring dashboard** (Grafana) for model performance
- [ ] Add **SHAP values** for model explainability

### MLOps
- [ ] Set up **CI/CD pipeline** with GitHub Actions
- [ ] Implement **automated model retraining** on new fraud patterns
- [ ] Add **model versioning** and rollback capability
- [ ] Create **Docker container** for easy deployment

### Feature Engineering
- [ ] Add **graph features** (customer-terminal transaction networks)
- [ ] Implement **sequence models** (LSTM) for transaction patterns
- [ ] Include **geographical features** (location-based risk)

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **ML/Data Science** | scikit-learn, XGBoost, LightGBM, pandas, NumPy |
| **API** | FastAPI, Pydantic, Uvicorn |
| **Database** | PostgreSQL, Supabase |
| **Testing** | pytest, requests |
| **Visualization** | Matplotlib, Seaborn |
| **Deployment** | Docker (planned), AWS (planned) |

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Dataset from [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/) by Le Borgne et al.
- Original Dataset in notebook found : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data
- Inspired by production fraud detection systems
- Thanks to the open-source ML community for excellent tools and libraries
---