from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.config import Settings, TargetMode, load_settings
from src.data.load_repository import LoadRepository
from src.evaluation.metrics import evaluate
from src.features.builder import FeatureBuilder
from src.models.lightgbm_model import LightGbmForecaster
from src.utils.weather_io import load_weather


def _required_target_columns(settings: Settings) -> list[str]:
    if settings.target_mode == TargetMode.RESIDUAL:
        return ["actual_mw", "tso_mw"]
    return ["actual_mw"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/settings.json")
    args = parser.parse_args()

    settings = load_settings(args.settings)

    if settings.holdout_days <= 0:
        raise ValueError("holdout_days must be positive for backtesting")

    repository = LoadRepository(settings)
    hourly = repository.load_hourly()

    weather = load_weather(settings, hourly)

    builder = FeatureBuilder(settings)
    dataset = builder.build(hourly, weather)

    required_columns = _required_target_columns(settings)
    valid_dataset = dataset.dropna(subset=required_columns)

    unique_dates = np.sort(pd.Series(valid_dataset.index.normalize()).unique())

    if len(unique_dates) <= settings.holdout_days:
        raise ValueError("not enough valid dates for holdout evaluation")

    test_dates = set(unique_dates[-settings.holdout_days :])
    test_mask = valid_dataset.index.normalize().isin(test_dates)
    test = valid_dataset[test_mask]

    if test.empty:
        raise ValueError("holdout frame is empty")

    forecaster = LightGbmForecaster.load(settings.artifact_path)
    predictions = forecaster.predict(test).to_numpy(dtype="float64")

    if settings.target_mode == TargetMode.RESIDUAL:
        forecast_values = test["tso_mw"].to_numpy(dtype="float64") + predictions
    else:
        forecast_values = predictions

    naive_values = None
    if "actual_lag_168h" in test.columns:
        naive_values = test["actual_lag_168h"].to_numpy(dtype="float64")

    report = evaluate(
        actual=test["actual_mw"].to_numpy(dtype="float64"),
        forecast=forecast_values,
        tso_forecast=test["tso_mw"].to_numpy(dtype="float64"),
        naive_forecast=naive_values,
    )

    print(report.model_dump_json(indent=2))


if __name__ == "__main__":
    main()