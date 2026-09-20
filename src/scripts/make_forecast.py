from __future__ import annotations

import argparse
from datetime import date

import pandas as pd

from src.config import Settings, TargetMode, load_settings
from src.data.load_repository import LoadRepository
from src.features.builder import FeatureBuilder
from src.forecast.pipeline import ForecastPipeline
from src.models.lightgbm_model import LightGbmForecaster
from src.utils.io import write_forecast_csv
from src.utils.weather_io import load_weather


def _default_target_date(dataset: pd.DataFrame, settings: Settings) -> date:
    if settings.target_mode == TargetMode.RESIDUAL:
        daily = dataset.groupby(dataset.index.normalize()).agg(
            row_count=("tso_mw", "size"),
            tso_count=("tso_mw", "count"),
        )

        valid_dates = daily[daily["row_count"] == daily["tso_count"]].index

        if valid_dates.empty:
            raise ValueError("no target date has a complete TSO forecast")

        return valid_dates.max().date()

    return dataset.index.normalize().max().date()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/settings.json")
    parser.add_argument("--target-date", default=None)
    args = parser.parse_args()

    settings = load_settings(args.settings)

    repository = LoadRepository(settings)
    hourly = repository.load_hourly()

    weather = load_weather(settings, hourly)

    builder = FeatureBuilder(settings)
    dataset = builder.build(hourly, weather)

    if dataset.empty:
        raise ValueError("dataset is empty")

    if args.target_date is None:
        target_date = _default_target_date(dataset, settings)
    else:
        target_date = pd.Timestamp(args.target_date).date()

    forecaster = LightGbmForecaster.load(settings.artifact_path)
    pipeline = ForecastPipeline(settings=settings, model=forecaster)

    rows = pipeline.predict_date(dataset, target_date)
    write_forecast_csv(rows, settings.output_path)

    print(f"wrote {len(rows)} rows to {settings.output_path}")


if __name__ == "__main__":
    main()