"""FastAPI service that scores card transactions for fraud."""

import uuid
from datetime import datetime
from typing import List, Optional

import joblib
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config.config import get_settings
from config.logging_config import get_logger, set_request_id, setup_logging
from src.feature_store.store import FeatureStore
from src.model_utils import score_transaction

settings = get_settings()

setup_logging(
    log_level=settings.LOG_LEVEL, log_format=settings.LOG_FORMAT, enable_colors=settings.DEBUG
)
logger = get_logger(__name__)

try:
    artifact = joblib.load(settings.MODEL_PATH)
    MODEL = artifact["model"]
    FEATURE_ORDER = artifact["feature_columns"]
    THRESHOLD = (
        settings.MODEL_THRESHOLD if settings.MODEL_THRESHOLD else float(artifact["threshold"])
    )
    MODEL_VERSION = artifact["model_type"]
    logger.info(
        "Model loaded successfully",
        extra={
            "model_version": MODEL_VERSION,
            "threshold": THRESHOLD,
            "num_features": len(FEATURE_ORDER),
            "model_path": settings.MODEL_PATH,
        },
    )
except Exception as e:
    logger.error(f"Failed to load model from {settings.MODEL_PATH}: {e}")
    raise

feature_store = FeatureStore()

app = FastAPI(
    title=settings.APP_NAME,
    description="Detect fraudulent credit card transactions in real-time using ML + Feature Store",
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_request_id_middleware(request: Request, call_next):
    """Add unique request ID to each request for tracing"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    set_request_id(request_id)

    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id

    return response


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
                "timestamp": "2024-01-15T14:30:00",
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
    """Log the transaction after the response is sent."""
    try:
        transaction = {
            "transaction_id": prediction["transaction_id"],
            "customer_id": tx_data["customer_id"],
            "terminal_id": tx_data["terminal_id"],
            "amount": tx_data["amount"],
            "timestamp": tx_data["timestamp"],
            # is_fraud is the confirmed label, set later from chargebacks or investigations.
            # Writing the model's own prediction here would feed it back into terminal fraud rates.
            "is_fraud": False,
            "fraud_probability": prediction["fraud_probability"],
        }

        feature_store.log_transaction(transaction)
        logger.info(f"Logged transaction {prediction['transaction_id']}")

    except Exception as e:
        logger.error(f"Error logging transaction: {e}")


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
        "model": MODEL_VERSION,
        "threshold": THRESHOLD,
        "version": settings.APP_VERSION,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(tx: Transaction, background_tasks: BackgroundTasks):
    """Predict fraud probability for a single transaction."""
    start_time = datetime.now()

    try:
        logger.info(
            "Processing transaction",
            extra={
                "transaction_id": tx.transaction_id or "unknown",
                "customer_id": tx.customer_id,
                "amount": tx.amount,
            },
        )

        tx_dict = tx.model_dump()

        result = score_transaction(
            tx=tx_dict,
            model=MODEL,
            feature_order=FEATURE_ORDER,
            threshold=THRESHOLD,
            model_version=MODEL_VERSION,
            feature_store=feature_store,
        )

        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        result["processing_time_ms"] = round(processing_time, 2)

        logger.info(
            "Prediction completed",
            extra={
                "transaction_id": result["transaction_id"],
                "fraud_probability": result["fraud_probability"],
                "is_fraud": result["is_fraud"],
                "risk_level": result["risk_level"],
                "processing_time_ms": processing_time,
            },
        )

        background_tasks.add_task(log_transaction_async, tx_dict, result)

        return result

    except ValueError as e:
        logger.error(
            f"Validation error: {e}",
            extra={"transaction": tx_dict if "tx_dict" in locals() else None},
        )
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")


@app.post("/batch-predict")
async def batch_predict(transactions: List[Transaction]):
    """Batch prediction for multiple transactions."""
    results = []
    error_count = 0

    logger.info(f"Processing batch of {len(transactions)} transactions")

    for tx in transactions:
        try:
            tx_dict = tx.model_dump()
            result = score_transaction(
                tx=tx_dict,
                model=MODEL,
                feature_order=FEATURE_ORDER,
                threshold=THRESHOLD,
                model_version=MODEL_VERSION,
                feature_store=feature_store,
            )
            results.append(result)
        except Exception as e:
            error_count += 1
            logger.error(f"Error processing transaction {tx.transaction_id}: {e}")
            results.append({"transaction_id": tx.transaction_id or "unknown", "error": str(e)})

    logger.info(
        "Batch processing completed",
        extra={
            "total": len(transactions),
            "successful": len(transactions) - error_count,
            "errors": error_count,
        },
    )

    return {"predictions": results, "count": len(results), "errors": error_count}


@app.get("/health")
def health():
    """Detailed health check"""
    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_type": MODEL_VERSION,
        "threshold": THRESHOLD,
        "num_features": len(FEATURE_ORDER),
        "feature_store": "connected",
    }


@app.get("/stats")
def stats():
    """Get system statistics"""
    try:
        customer_data = (
            feature_store.client.table("customer_features").select("customer_id").execute()
        )
        terminal_data = (
            feature_store.client.table("terminal_features").select("terminal_id").execute()
        )

        return {
            "customers_in_store": len(customer_data.data),
            "terminals_in_store": len(terminal_data.data),
            "model_version": MODEL_VERSION,
            "threshold": THRESHOLD,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching stats: {e}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        log_level=settings.LOG_LEVEL.lower(),
        workers=1,  # Use workers=1 for development, settings.API_WORKERS for production
    )
