from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from src.config import Objective, TargetMode


class ForecastRow(BaseModel):
    forecast_date: date
    local_hour: int = Field(ge=0, le=23)
    forecast_mw: float
    tso_forecast_mw: float | None = None
    predicted_residual_mw: float | None = None


class BacktestReport(BaseModel):
    row_count: int
    mae_mw: float
    rmse_mw: float
    bias_mw: float
    mae_pct_of_mean_actual: float | None = None
    tso_mae_mw: float | None = None
    naive_last_week_mae_mw: float | None = None


class TuningResult(BaseModel):
    best_validation_mae: float
    best_params: dict[str, Any]
    best_iteration: int | None = None


class ModelMetadata(BaseModel):
    feature_names: list[str]
    target_mode: TargetMode
    objective: Objective
    quantile_tau: float
    params: dict[str, Any]
    created_at: datetime