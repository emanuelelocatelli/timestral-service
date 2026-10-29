"""Timestral inference and the same direction/confidence reading Chronos exposes.

The direction rule is copied from the Chronos service so a stored UP/DOWN/NEUTRAL
means the same thing: last forecast versus last close, with a threshold of half
the return standard deviation, clamped to [0.05, 2.0] percent.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

BAND_LOW = 0.1
BAND_HIGH = 0.9
MEDIAN = 0.5


@dataclass
class PredictionResult:
    forecast: List[float]
    forecast_low: List[float]
    forecast_high: List[float]
    prediction_direction: str
    trend_strength: str
    confidence_score: float
    volatility: float
    context_length_used: int
    interval_seconds: int
    quantiles: dict = field(default_factory=dict)
    quantile_levels: List[float] = field(default_factory=list)
    mean: List[float] = field(default_factory=list)
    inference_ms: float = 0.0
    label: Optional[str] = None


class TimestralPredictor:
    def __init__(self) -> None:
        self.model_name = "mistralai/Timestral-209M"
        self.requested_device = "cpu"
        self.device = "cpu"
        self.torch_num_threads = 4
        self.forecaster = None
        self.is_loaded = False
        self._lock = threading.Lock()
        self._forecast_batch: Optional[Callable[[np.ndarray, int], Tuple[np.ndarray, Tuple[float, ...]]]] = None

    def load_model(self) -> None:
        from timestral import TimestralForecaster

        self._apply_threads()
        self.device = self._resolve_device(self.requested_device)
        logger.info("Loading Timestral '%s' on %s", self.model_name, self.device)
        self.forecaster = TimestralForecaster.from_pretrained(self.model_name, device=self.device)
        self.is_loaded = True
        logger.info("Timestral loaded on %s", self.device)

    def describe(self) -> dict:
        return {
            "model_name": self.model_name,
            "device": self.device,
            "model_loaded": self.is_loaded,
            "torch_num_threads": self.torch_num_threads,
            "context_length": 32768,
            "quantile_levels": list(getattr(self.forecaster, "quantile_levels", ()) or ()),
        }

    def predict(
        self,
        context_data: Sequence[float],
        prediction_horizon: int,
        interval_seconds: int,
    ) -> PredictionResult:
        return self.predict_many([list(context_data)], prediction_horizon, interval_seconds)[0]

    def predict_many(
        self,
        contexts: Sequence[Sequence[float]],
        prediction_horizon: int,
        interval_seconds: int,
        labels: Optional[Sequence[Optional[str]]] = None,
    ) -> List[PredictionResult]:
        if not contexts:
            return []

        grouped: dict[int, List[int]] = {}
        for index, context in enumerate(contexts):
            grouped.setdefault(len(context), []).append(index)

        ordered: List[Optional[PredictionResult]] = [None] * len(contexts)
        started = time.monotonic()

        for indexes in grouped.values():
            batch = np.stack([np.asarray(contexts[index], dtype=np.float64) for index in indexes])
            quantiles, levels = self._run_batch(batch, prediction_horizon)
            for row, index in enumerate(indexes):
                result = self._to_result(
                    np.asarray(contexts[index], dtype=np.float64),
                    quantiles[row],
                    levels,
                    prediction_horizon,
                    interval_seconds,
                )
                result.label = None if labels is None else labels[index]
                ordered[index] = result

        elapsed = (time.monotonic() - started) * 1000.0
        per_series = elapsed / len(contexts)
        results: List[PredictionResult] = []
        for item in ordered:
            if item is None:
                raise RuntimeError("Timestral batch left a series without a forecast")
            item.inference_ms = round(per_series, 3)
            results.append(item)
        return results

    def _run_batch(self, batch: np.ndarray, horizon: int) -> Tuple[np.ndarray, Tuple[float, ...]]:
        if self._forecast_batch is not None:
            return self._forecast_batch(batch, horizon)
        if not self.is_loaded or self.forecaster is None:
            raise RuntimeError("Model not loaded")

        with self._lock:
            forecast = self.forecaster.forecast(batch, prediction_length=horizon)

        quantiles = forecast.quantiles.detach().to("cpu").numpy()
        if quantiles.ndim != 4:
            raise RuntimeError(f"Unexpected Timestral quantile shape {quantiles.shape}")
        # (series, variates, horizon, quantile). One target variate.
        return quantiles[:, 0, :, :], tuple(float(level) for level in forecast.quantile_levels)

    def _to_result(
        self,
        context: np.ndarray,
        quantile_matrix: np.ndarray,
        levels: Sequence[float],
        horizon: int,
        interval_seconds: int,
    ) -> PredictionResult:
        low_index = _level_index(levels, BAND_LOW)
        median_index = _level_index(levels, MEDIAN)
        high_index = _level_index(levels, BAND_HIGH)
        forecast_low = quantile_matrix[:, low_index]
        forecast_median = quantile_matrix[:, median_index]
        forecast_high = quantile_matrix[:, high_index]
        last_close = float(context[-1])
        direction, strength = direction_and_strength(last_close, float(forecast_median[-1]), context)
        return PredictionResult(
            forecast=_round_list(forecast_median),
            forecast_low=_round_list(forecast_low),
            forecast_high=_round_list(forecast_high),
            prediction_direction=direction,
            trend_strength=strength,
            confidence_score=confidence_score(
                forecast_median, forecast_low, forecast_high, last_close, horizon
            ),
            volatility=volatility(context),
            context_length_used=int(context.shape[0]),
            interval_seconds=interval_seconds,
            quantiles={
                _quantile_key(level): _round_list(quantile_matrix[:, index])
                for index, level in enumerate(levels)
            },
            quantile_levels=[float(level) for level in levels],
            mean=_round_list(_quantile_mean(quantile_matrix, levels)),
        )

    def _apply_threads(self) -> None:
        if self.torch_num_threads < 1:
            return
        import torch

        torch.set_num_threads(int(self.torch_num_threads))

    @staticmethod
    def _resolve_device(requested: str) -> str:
        normalized = (requested or "cpu").strip().lower()
        if normalized == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        if normalized == "cuda":
            import torch

            if not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, using CPU")
                return "cpu"
        return normalized


def direction_and_strength(
    last_context: float,
    last_forecast: float,
    context: np.ndarray,
) -> Tuple[str, str]:
    if abs(last_context) < 1e-12:
        return "NEUTRAL", "none"

    percent_change = ((last_forecast - last_context) / abs(last_context)) * 100.0
    returns = _returns(context)
    threshold = 0.5 * float(np.std(returns)) * 100.0 if len(returns) > 1 else 0.1
    threshold = max(0.05, min(threshold, 2.0))

    if percent_change > threshold:
        direction = "UP"
    elif percent_change < -threshold:
        direction = "DOWN"
    else:
        direction = "NEUTRAL"

    ratio = abs(percent_change) / threshold if threshold > 0 else 0.0
    if ratio > 3.0:
        strength = "strong"
    elif ratio > 1.5:
        strength = "moderate"
    elif ratio > 1.0:
        strength = "weak"
    else:
        strength = "none"
    return direction, strength


def confidence_score(
    forecast_median: np.ndarray,
    forecast_low: np.ndarray,
    forecast_high: np.ndarray,
    last_context: float,
    horizon: int,
) -> float:
    if abs(last_context) < 1e-12:
        return 0.5
    spreads = np.abs(forecast_high - forecast_low) / abs(last_context)
    spread_confidence = max(0.0, 1.0 - float(np.mean(spreads)))
    horizon_penalty = max(0.0, 1.0 - (horizon / 50.0))
    raw = 0.75 * spread_confidence + 0.25 * horizon_penalty
    return round(max(0.10, min(raw, 0.95)), 4)


def volatility(context: np.ndarray) -> float:
    returns = _returns(context)
    if len(returns) > 1:
        return round(float(np.std(returns)) * np.sqrt(252), 6)
    return 0.0


def _returns(data: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        returns = np.diff(data) / np.abs(data[:-1])
    return returns[np.isfinite(returns)]


def _level_index(levels: Sequence[float], target: float) -> int:
    for index, level in enumerate(levels):
        if abs(float(level) - target) < 1e-9:
            return index
    raise ValueError(f"Timestral quantile levels do not include {target}: {levels}")


def _quantile_key(level: float) -> str:
    return f"q{int(round(float(level) * 100))}"


def _quantile_mean(quantile_matrix: np.ndarray, levels: Sequence[float]) -> np.ndarray:
    """Integrate the quantile function so `mean` is not a copy of the median."""
    grid = np.asarray(levels, dtype=np.float64)
    if grid.size < 2 or float(grid[-1] - grid[0]) <= 0:
        return quantile_matrix[:, quantile_matrix.shape[1] // 2]
    span = float(grid[-1] - grid[0])
    deltas = np.diff(grid)
    slabs = (quantile_matrix[:, :-1] + quantile_matrix[:, 1:]) * 0.5 * deltas
    return np.sum(slabs, axis=1) / span


def _round_list(values: np.ndarray, decimals: int = 6) -> List[float]:
    return [round(float(value), decimals) for value in np.asarray(values).reshape(-1)]
