from __future__ import annotations

import argparse

from src.config import load_settings
from src.data.load_repository import LoadRepository
from src.features.builder import FeatureBuilder
from src.tuning.optuna_tuner import OptunaTuner
from src.utils.weather_io import load_weather


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/settings.json")
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    settings = load_settings(args.settings)

    repository = LoadRepository(settings)
    hourly = repository.load_hourly()

    weather = load_weather(settings, hourly)

    builder = FeatureBuilder(settings)
    dataset = builder.build(hourly, weather)
    feature_names = builder.feature_columns(dataset)

    tuner = OptunaTuner(settings=settings, feature_names=feature_names)
    result = tuner.tune(dataset, n_trials=args.trials)

    forecaster = tuner.build_final_forecaster(dataset, result)
    forecaster.save(settings.artifact_path)

    print(f"best validation mae {result.best_validation_mae:.3f}")
    print(f"saved artifact to {settings.artifact_path}")


if __name__ == "__main__":
    main()