from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.domain.schemas import BacktestReport


def evaluate(
    actual: np.ndarray,
    forecast: np.ndarray,
    tso_forecast: np.ndarray | None = None,
    naive_forecast: np.ndarray | None = None,
) -> BacktestReport:
    actual = np.asarray(actual, dtype="float64")
    forecast = np.asarray(forecast, dtype="float64")

    mask = np.isfinite(actual) & np.isfinite(forecast)

    if mask.sum() == 0:
        raise ValueError("no valid rows to evaluate")

    actual = actual[mask]
    forecast = forecast[mask]

    errors = forecast - actual

    mae = float(mean_absolute_error(actual, forecast))
    rmse = float(np.sqrt(mean_squared_error(actual, forecast)))
    bias = float(np.mean(errors))

    mean_absolute_actual = float(np.mean(np.abs(actual)))
    mae_pct = None
    if mean_absolute_actual > 0:
        mae_pct = 100.0 * mae / mean_absolute_actual

    tso_mae = None
    if tso_forecast is not None:
        tso = np.asarray(tso_forecast, dtype="float64")[mask]
        tso_mask = np.isfinite(tso)
        if tso_mask.sum() > 0:
            tso_mae = float(mean_absolute_error(actual[tso_mask], tso[tso_mask]))

    naive_mae = None
    if naive_forecast is not None:
        naive = np.asarray(naive_forecast, dtype="float64")[mask]
        naive_mask = np.isfinite(naive)
        if naive_mask.sum() > 0:
            naive_mae = float(mean_absolute_error(actual[naive_mask], naive[naive_mask]))

    return BacktestReport(
        row_count=int(mask.sum()),
        mae_mw=mae,
        rmse_mw=rmse,
        bias_mw=bias,
        mae_pct_of_mean_actual=mae_pct,
        tso_mae_mw=tso_mae,
        naive_last_week_mae_mw=naive_mae,
    )