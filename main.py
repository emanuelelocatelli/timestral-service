import logging
import math
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from config import get_settings
from engine import PredictionResult, TimestralPredictor

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

predictor = TimestralPredictor()
predictor.model_name = settings.model_name
predictor.requested_device = settings.device
predictor.torch_num_threads = settings.torch_num_threads


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load_model()
    yield
    logger.info("Shutting down Timestral service")


app = FastAPI(
    title="Timestral Forecasting Service",
    version=settings.service_version,
    lifespan=lifespan,
)


@app.middleware("http")
async def require_token(request: Request, call_next):
    token = (get_settings().api_token or "").strip()
    if token and request.url.path.startswith("/predict"):
        header = request.headers.get("authorization", "")
        presented = header[7:].strip() if header.lower().startswith("bearer ") else ""
        if presented != token:
            return JSONResponse(status_code=401, content={"detail": "Invalid or missing bearer token"})
    return await call_next(request)


class PredictRequest(BaseModel):
    pair: str = Field(..., min_length=1, max_length=50)
    context_data: List[float] = Field(..., min_length=8)
    prediction_horizon: Optional[int] = Field(default=None, ge=1, le=1024)
    interval_seconds: Optional[int] = Field(default=None, ge=1, le=31_536_000)
    timestamp_kline: Optional[str] = None
    candle_id: Optional[int] = None
    sequence_number: Optional[int] = None
    timestamp_ms: Optional[int] = None
    request_id: Optional[str] = None

    @field_validator("context_data")
    @classmethod
    def finite_context(cls, values: List[float]) -> List[float]:
        for index, value in enumerate(values):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"context_data[{index}] contains NaN or Inf")
        return values


class PredictionResponse(BaseModel):
    pair: str
    prediction_direction: str
    trend_strength: str
    confidence_score: float
    forecast: List[float]
    forecast_low: List[float]
    forecast_high: List[float]
    volatility: float
    predicted_at: str
    prediction_valid_seconds: int
    context_length_used: int
    interval_seconds: int
    quantiles_10: List[float]
    quantiles_90: List[float]
    quantiles: Dict[str, List[float]]
    mean: List[float]


class BatchSeries(BaseModel):
    key: str = Field(..., min_length=1, max_length=120)
    context_data: List[float] = Field(..., min_length=8)

    @field_validator("context_data")
    @classmethod
    def finite_context(cls, values: List[float]) -> List[float]:
        for index, value in enumerate(values):
            if math.isnan(value) or math.isinf(value):
                raise ValueError(f"context_data[{index}] contains NaN or Inf")
        return values


class PredictBatchRequest(BaseModel):
    series: List[BatchSeries] = Field(..., min_length=1)
    prediction_horizon: Optional[int] = Field(default=None, ge=1, le=1024)
    interval_seconds: Optional[int] = Field(default=None, ge=1, le=31_536_000)


class BatchPredictionItem(BaseModel):
    key: str
    prediction_direction: str
    trend_strength: str
    confidence_score: float
    forecast: List[float]
    forecast_low: List[float]
    forecast_high: List[float]
    quantiles_10: List[float]
    quantiles_90: List[float]
    quantiles: Dict[str, List[float]]
    mean: List[float]
    volatility: float
    context_length_used: int
    interval_seconds: int
    prediction_valid_seconds: int


class PredictBatchResponse(BaseModel):
    count: int
    predicted_at: str
    total_inference_ms: float
    ms_per_series: float
    results: List[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str
    device: str
    model_loaded: bool
    torch_num_threads: int


def _response(pair: str, result: PredictionResult, horizon: int) -> PredictionResponse:
    return PredictionResponse(
        pair=pair,
        prediction_direction=result.prediction_direction,
        trend_strength=result.trend_strength,
        confidence_score=result.confidence_score,
        forecast=result.forecast,
        forecast_low=result.forecast_low,
        forecast_high=result.forecast_high,
        volatility=result.volatility,
        predicted_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        prediction_valid_seconds=max(1, result.interval_seconds * horizon),
        context_length_used=result.context_length_used,
        interval_seconds=result.interval_seconds,
        quantiles_10=result.forecast_low,
        quantiles_90=result.forecast_high,
        quantiles=result.quantiles,
        mean=result.mean,
    )


def _batch_item(key: str, result: PredictionResult, horizon: int) -> BatchPredictionItem:
    return BatchPredictionItem(
        key=key,
        prediction_direction=result.prediction_direction,
        trend_strength=result.trend_strength,
        confidence_score=result.confidence_score,
        forecast=result.forecast,
        forecast_low=result.forecast_low,
        forecast_high=result.forecast_high,
        quantiles_10=result.forecast_low,
        quantiles_90=result.forecast_high,
        quantiles=result.quantiles,
        mean=result.mean,
        volatility=result.volatility,
        context_length_used=result.context_length_used,
        interval_seconds=result.interval_seconds,
        prediction_valid_seconds=max(1, result.interval_seconds * horizon),
    )


@app.get("/", response_model=HealthResponse)
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    described = predictor.describe()
    loaded = described["model_loaded"]
    if not loaded:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="Timestral Forecasting Service is running (model loaded)",
        version=get_settings().service_version,
        model=described["model_name"],
        device=described["device"],
        model_loaded=True,
        torch_num_threads=described["torch_num_threads"],
    )


@app.post("/predict", response_model=PredictionResponse)
def predict(body: PredictRequest) -> PredictionResponse:
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    horizon = body.prediction_horizon or get_settings().default_prediction_horizon
    interval = body.interval_seconds or get_settings().default_interval_seconds
    try:
        result = predictor.predict(body.context_data, horizon, interval)
    except Exception as exception:
        logger.exception("Prediction failed for %s", body.pair)
        raise HTTPException(status_code=500, detail=f"Prediction error: {exception}") from exception
    return _response(body.pair, result, horizon)


@app.post("/predict-batch", response_model=PredictBatchResponse)
def predict_batch(body: PredictBatchRequest) -> PredictBatchResponse:
    if not predictor.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    limit = get_settings().max_batch_series
    if len(body.series) > limit:
        raise HTTPException(status_code=422, detail=f"{len(body.series)} series requested, limit is {limit}")

    keys = [item.key for item in body.series]
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        raise HTTPException(status_code=422, detail=f"duplicate series keys: {duplicates}")

    horizon = body.prediction_horizon or get_settings().default_prediction_horizon
    interval = body.interval_seconds or get_settings().default_interval_seconds
    started = datetime.now(timezone.utc)
    try:
        results = predictor.predict_many(
            [item.context_data for item in body.series],
            horizon,
            interval,
            labels=keys,
        )
    except Exception as exception:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {exception}") from exception

    total_ms = sum(item.inference_ms for item in results)
    return PredictBatchResponse(
        count=len(results),
        predicted_at=started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        total_inference_ms=round(total_ms, 3),
        ms_per_series=round(total_ms / max(1, len(results)), 3),
        results=[_batch_item(item.key, result, horizon) for item, result in zip(body.series, results)],
    )
