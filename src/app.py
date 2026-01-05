from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import joblib
from typing import Optional
import logging
from datetime import datetime

from model_utils import score_transaction
from feature_store.store import FeatureStore

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load model
try:
    artifact = joblib.load("../artifacts/fraud_detection_model_final.pkl")
    MODEL = artifact["model"]
    FEATURE_ORDER = artifact["feature_columns"]
    THRESHOLD = float(artifact["threshold"])
    MODEL_VERSION = artifact["model_type"]
    logger.info(f"Model loaded: {MODEL_VERSION}")
    logger.info(f"Threshold: {THRESHOLD:.4f}")
    logger.info(f"Features: {len(FEATURE_ORDER)}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Initialize feature store
feature_store = FeatureStore()

app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Detect fraudulent credit card transactions in real-time using ML + Feature Store",
    version="2.0.0"
)


class Transaction(BaseModel):
    transaction_id: Optional[str] = None
    customer_id: int = Field(..., description="Customer ID", ge=0)
    terminal_id: int = Field(..., description="Terminal/Merchant ID", ge=0)
    amount: float = Field(..., description="Transaction amount", ge=0)
    timestamp: str = Field(..., description="ISO format timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "tx_12345",
                "customer_id": 1234,
                "terminal_id": 5678,
                "amount": 125.50,
                "timestamp": "2024-01-15T14:30:00"
            }
        }


class PredictionResponse(BaseModel):
    transaction_id: str
    customer_id: int
    terminal_id: int
    amount: float
    fraud_probability: float
    is_fraud: bool
    risk_level: str
    threshold: float
    model_version: str
    timestamp: str
    features_used: dict
    processing_time_ms: Optional[float] = None


def log_transaction_async(tx_data: dict, prediction: dict):
    """
    Background task to log transaction and update feature store.
    This runs AFTER the response is sent to the user.
    """
    try:
        # Log to transaction history
        transaction = {
            "transaction_id": prediction["transaction_id"],
            "customer_id": tx_data["customer_id"],
            "terminal_id": tx_data["terminal_id"],
            "amount": tx_data["amount"],
            "timestamp": tx_data["timestamp"],
            "is_fraud": prediction["is_fraud"],
            "fraud_probability": prediction["fraud_probability"]
        }
        
        feature_store.log_transaction(transaction)
        logger.info(f"Logged transaction {prediction['transaction_id']}")
        
    except Exception as e:
        logger.error(f"Error logging transaction: {e}")


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Fraud Detection API",
        "model": MODEL_VERSION,
        "threshold": THRESHOLD,
        "version": "2.0.0"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(tx: Transaction, background_tasks: BackgroundTasks):
    """
    Predict fraud probability for a transaction.
    
    
    Returns:
    - fraud_probability: 0.0 to 1.0
    - is_fraud: boolean decision
    - risk_level: CRITICAL/HIGH/MEDIUM/LOW
    - features_used: Key features that influenced the decision
    """
    start_time = datetime.now()
    
    try:
        logger.info(f"Processing transaction: {tx.transaction_id or 'unknown'}")
        
        tx_dict = tx.model_dump()
        
        result = score_transaction(
            tx=tx_dict,
            model=MODEL,
            feature_order=FEATURE_ORDER,
            threshold=THRESHOLD,
            model_version=MODEL_VERSION,
            feature_store=feature_store
        )
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000 
        result["processing_time_ms"] = round(processing_time, 2)
        
        logger.info(
            f"Prediction: {result['fraud_probability']:.4f} | "
            f"Fraud: {result['is_fraud']} | "
            f"Time: {processing_time:.2f}ms"
        )
        
        # Log transaction
        background_tasks.add_task(log_transaction_async, tx_dict, result)
        
        return result
    
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch-predict")
async def batch_predict(transactions: list[Transaction]):
    """
    Batch prediction for multiple transactions.
    Useful for processing queued transactions.
    """
    results = []
    
    for tx in transactions:
        try:
            tx_dict = tx.model_dump()
            result = score_transaction(
                tx=tx_dict,
                model=MODEL,
                feature_order=FEATURE_ORDER,
                threshold=THRESHOLD,
                model_version=MODEL_VERSION,
                feature_store=feature_store
            )
            results.append(result)
        except Exception as e:
            logger.error(f"Error processing transaction {tx.transaction_id}: {e}")
            results.append({
                "transaction_id": tx.transaction_id,
                "error": str(e)
            })
    
    return {"predictions": results, "count": len(results)}


@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_type": MODEL_VERSION,
        "threshold": THRESHOLD,
        "num_features": len(FEATURE_ORDER),
        "feature_store": "connected"
    }


@app.get("/stats")
def stats():
    """Get system statistics"""
    try:
        customer_data = feature_store.client.table('customer_features').select('customer_id').execute()
        terminal_data = feature_store.client.table('terminal_features').select('terminal_id').execute()
        
        return {
            "customers_in_store": len(customer_data.data),
            "terminals_in_store": len(terminal_data.data),
            "model_version": MODEL_VERSION,
            "threshold": THRESHOLD
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")