# 🛡️ Real-Time Fraud Detection System

An end-to-end machine learning system for detecting fraudulent credit card transactions in real-time, catching 75% of fraud while sending only 1% of transactions to review, evaluated with realistic 7-day label delays.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## 🎯 Project Overview

This project demonstrates a production-ready fraud detection system built from scratch, covering the complete ML lifecycle from data exploration to API deployment. The system processes transactions in real-time, leveraging historical customer and terminal behavior to identify fraudulent patterns.

### Key Features
-  **Real-time predictions** in under 1 second
-  **75.3% fraud recall at 65.3% precision**, flagging 1% of transactions
-  **Leakage-free features**: calendar windows and delayed fraud labels, shared by training and the live feature store
-  **Feature store architecture** for low-latency inference
-  **Asynchronous background processing** for feature updates
-  **Comprehensive model evaluation** with proper time-series validation


## 📈 Model Performance

Test period (days 155–182). Each threshold is chosen on validation to flag 1% of transactions, then applied unchanged to test. Reproduce with `python scripts/train.py`; results shift by about a point across XGBoost/scikit-learn versions.

| Model | ROC-AUC | PR-AUC | Recall | Precision | Alert Rate | False Alarms |
|-------|---------|--------|--------|-----------|------------|--------------|
| **XGBoost** | 0.902 | **0.735** | **75.3%** | **65.3%** | 1.02% | **955** |
| Random Forest | 0.905 | 0.703 | 72.3% | 60.9% | 1.05% | 1,107 |
| LightGBM | 0.899 | 0.624 | 69.7% | 61.7% | 1.00% | 1,030 |
| Logistic Regression | 0.905 | 0.501 | 63.2% | 54.8% | 1.02% | 1,241 |

XGBoost recall by fraud scenario: **96.3%** high-amount fraud, **68.7%** compromised terminals, **85.4%** compromised cards.

### Business Impact
- **1,793 of 2,382 frauds caught** while reviewing about 1% of transactions
- **955 false alarms**: roughly 2 in 3 alerts are real fraud
- Compromised terminals are the hardest case: their fraud only becomes visible once labels arrive

> **Earlier numbers were inflated.** A previous version reported 90.2% recall at 83.3% precision. Terminal fraud rates used the label of the *immediately previous* transaction, which is not known in practice. See [Key Learnings](#-key-learnings--challenges).

---

## 🚀 Getting Started

```bash
./setup.sh                        # creates .venv, installs dependencies, copies .env.example to .env
source .venv/bin/activate
# Edit .env with your SUPABASE_URL and SUPABASE_KEY
```

Generate the dataset with the [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/) simulator and save it as `data/raw/transactions_full.csv`. Then:

```bash
python scripts/train.py --save            # train, evaluate, and write artifacts/fraud_detection_model_final.pkl
python src/initlialize/initialize.py      # load customer/terminal features into Supabase
uvicorn src.app:app --reload              # API on http://localhost:8000
pytest                                    # run the test suite
```

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
- Fraud rate, counting only transactions at least 7 days old (labels arrive late)
- Average transaction amount
- High-risk terminal flag (>5% fraud rate)

#### Interaction Features
- Night + high amount combination
- Combined risk score (weighted sum of risk indicators)

---

## 🔬 Methodology

### 1. Data Leakage Prevention
**Critical insight**: features may only use information known at prediction time, including *when labels become known*.
```python
# Wrong: "last 7 rows", and the previous transaction's fraud label is assumed to be known instantly
df['fraud_rate'] = df.groupby('terminal')['is_fraud'].transform(lambda x: x.shift(1).rolling(7).mean())

# Correct: calendar windows, counting only labels at least 7 days old
# fraud rate over [t - 14 days, t - 7 days); see src/features.py
```
`tests/unit/test_features.py` checks that recent fraud is not visible and that windows are in calendar days.

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

**Second problem**: Even after shifting, results were too good (90% recall at 83% precision)  
**Root Cause**: Terminal fraud rates used labels of transactions seconds old. In this dataset a compromised terminal is fraudulent for weeks, so "the last transaction here was fraud" nearly gave away the answer. Windows were also "last N rows" instead of "last N days", which didn't match the live feature store  
**Solution**: Calendar windows plus a 7-day label delay, applied in both training (`src/features.py`) and serving (`feature_store/store.py`). Recall at a 1% alert budget dropped to a realistic 75%

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