from __future__ import annotations

from datetime import date

import pandas as pd

from src.config import Settings, TargetMode
from src.domain.schemas import ForecastRow
from src.models.lightgbm_model import LightGbmForecaster


class ForecastPipeline:
    def __init__(self, settings: Settings, model: LightGbmForecaster) -> None:
        self._settings = settings
        self._model = model

    def predict_date(
        self,
        dataset: pd.DataFrame,
        target_date: date,
    ) -> list[ForecastRow]:
        rows = dataset[dataset.index.date == target_date]

        if rows.empty:
            raise ValueError(f"no dataset rows for target date {target_date}")

        if (
            self._settings.target_mode == TargetMode.RESIDUAL
            and rows["tso_mw"].isna().any()
        ):
            raise ValueError("residual forecasting requires TSO forecast values")

        predictions = self._model.predict(rows).to_numpy(dtype="float64")

        if self._settings.target_mode == TargetMode.RESIDUAL:
            forecast_values = rows["tso_mw"].to_numpy(dtype="float64") + predictions
            residual_values = predictions
        else:
            forecast_values = predictions
            residual_values = [None] * len(rows)

        records = []

        for position, index in enumerate(rows.index):
            tso_value = rows["tso_mw"].iloc[position]
            residual_value = residual_values[position]

            records.append(
                ForecastRow(
                    forecast_date=index.date(),
                    local_hour=index.hour,
                    forecast_mw=float(forecast_values[position]),
                    tso_forecast_mw=None if pd.isna(tso_value) else float(tso_value),
                    predicted_residual_mw=(
                        None
                        if residual_value is None or pd.isna(residual_value)
                        else float(residual_value)
                    ),
                )
            )

        return records