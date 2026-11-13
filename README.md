# Real-Time Fraud Detection

A FastAPI service that scores card transactions with XGBoost, using each customer's and terminal's recent history from a feature store. On held-out data it catches **76% of fraud while flagging 1% of transactions** for review, and about 2 in 3 alerts are real fraud.

## Results

Test period: the last 28 of 183 days. Each model's threshold is chosen on the validation period to flag 1% of transactions, then applied unchanged to test.

| Model | PR-AUC | Recall | Precision | False alarms |
|-------|--------|--------|-----------|--------------|
| **XGBoost** | **0.754** | **76.0%** | 65.8% | 939 |
| Random Forest | 0.722 | 76.1% | 64.5% | 1,000 |
| LightGBM | 0.656 | 75.2% | 66.7% | 896 |
| Logistic Regression | 0.524 | 66.5% | 55.6% | 1,264 |

By fraud type, XGBoost catches 98% of high-amount fraud, 88% of stolen-card fraud, and 69% of fraud at compromised terminals. Results move by about a point across XGBoost and scikit-learn versions; the committed model scored 76.7% recall at 66.0% precision.

## How it works

**Data.** 1.75M simulated card transactions over 183 days from the [Fraud Detection Handbook](https://fraud-detection-handbook.github.io/fraud-detection-handbook/) simulator, with about 5,000 customers and 10,000 terminals. 0.84% are fraud, from three patterns: purchases over $220, terminals compromised for 28 days, and stolen cards that spend about 5x the owner's usual amount for 14 days.

**Features** (`src/features.py`). 28 per transaction: time of day and week, the amount, the customer's spending over the previous 1, 7 and 30 days (including this purchase as a multiple of their average), and the terminal's volume and fraud rate over the same windows.

**Serving.** `POST /predict` reads the customer's and terminal's precomputed features from Supabase, builds the same 28 features, and scores the transaction in well under a second. The transaction is logged after the response is sent.

## Design decisions

- **Fraud labels arrive late.** A transaction's fraud label is only known days later, after a chargeback or investigation. Terminal fraud rates therefore count only transactions at least 7 days old. An earlier version used the previous transaction's label directly; that leak inflated recall to 91% at 81% precision.
- **Split by time, not at random.** Train on days 0 to 127, validate on 128 to 154, test on 155 to 182. A random split would let the model learn from the future.
- **Evaluate at a fixed review budget.** A fraud team can review a fixed share of transactions, so the question is how much fraud a 1% budget catches, not recall at an arbitrary threshold.
- **PR-AUC, not accuracy.** With 0.84% fraud, predicting "not fraud" every time is 99.2% accurate.
- **One definition of each feature.** Training, the loader and the API follow the same window and delay rules, and tests fail if serving stops producing a feature the model expects.

## Limitations

- **Live history isn't refreshed.** `FeatureStore` can recompute a customer's or terminal's features from `transaction_history`, but nothing calls it yet, so live traffic is logged without updating the stored features.
- **No way to record confirmed fraud.** New transactions are stored as not fraud. Without a feedback endpoint, terminal fraud rates fade to zero as the loaded history ages out.
- **Some fraud is undetectable here.** About a quarter of compromised-terminal fraud happens before any label at that terminal is confirmed, and those purchases look normal.
- **Simulated data.** Real fraud is messier than three planted patterns.

## Running it

Requires Python 3.9+ and a Supabase project with `customer_features`, `terminal_features` and `transaction_history` tables.

```bash
./setup.sh                    # virtualenv, dependencies, and .env from .env.example
source .venv/bin/activate     # then add SUPABASE_URL and SUPABASE_KEY to .env
```

Save the simulator's output as `data/raw/transactions_full.csv`, then:

```bash
python scripts/train.py --save        # train, evaluate, write artifacts/fraud_detection_model_final.pkl
python scripts/load_feature_store.py  # load features and history into Supabase
uvicorn src.app:app --reload          # API on http://localhost:8000
pytest                                # 105 tests
```

`docker compose -f docker/docker-compose.yml up` runs the API in a container. The `frontend/` folder is a small React form for trying predictions.

## Layout

```
src/features.py           feature definitions shared by training and tests
src/model_utils.py        builds features for one transaction and scores it
src/feature_store/        Supabase reads and writes
src/app.py                FastAPI service
scripts/train.py          trains and evaluates the four models
scripts/load_feature_store.py   loads the feature store from the CSV
notebooks/fraud.ipynb     exploration, including an earlier pass on the Kaggle dataset
tests/                    unit and API tests
```
