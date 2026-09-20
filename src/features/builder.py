from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import Settings
from src.features.calendar import add_calendar_features


class FeatureBuilder:
    target_columns = {"actual_mw", "tso_mw", "residual_mw"}

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def build(
        self,
        hourly: pd.DataFrame,
        weather: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        frame = hourly.sort_index().copy()

        out = add_calendar_features(frame.index)
        out["actual_mw"] = frame["actual_mw"]
        out["tso_mw"] = frame["tso_mw"]
        out["residual_mw"] = out["actual_mw"] - out["tso_mw"]
        
        volatility_cols = ["actual_std", "tso_std", "actual_range", "tso_range"]
        for col in volatility_cols:
            out[col] = frame[col] if col in frame else np.nan

        self._add_load_features(out, frame)
        self._add_tso_features(out, frame)
        self._add_error_features(out, frame)

        if self._settings.include_weather:
            self._add_weather_features(out, weather)

        return out

    def feature_columns(self, dataset: pd.DataFrame) -> list[str]:
        return [
            column
            for column in dataset.columns
            if column not in self.target_columns
            and pd.api.types.is_numeric_dtype(dataset[column])
        ]

    @staticmethod
    def _same_hour_shift(series: pd.Series, periods: int) -> pd.Series:
        return series.groupby(series.index.hour).transform(
            lambda group: group.shift(periods)
        )

    @staticmethod
    def _rolling_by_hour(
        series: pd.Series,
        window: int,
        aggregation: str,
        shift_periods: int = 2,
    ) -> pd.Series:
        shifted = series.groupby(series.index.hour).transform(
            lambda group: group.shift(shift_periods)
        )

        if aggregation == "mean":
            return shifted.groupby(shifted.index.hour).transform(
                lambda group: group.rolling(
                    window,
                    min_periods=max(1, window // 2),
                ).mean()
            )

        if aggregation == "std":
            return shifted.groupby(shifted.index.hour).transform(
                lambda group: group.rolling(
                    window,
                    min_periods=max(2, window // 2),
                ).std()
            )

        raise ValueError(f"unsupported aggregation {aggregation}")

    @staticmethod
    def _map_date_series(series: pd.Series, dates: pd.Index) -> np.ndarray:
        if series.empty:
            return np.full(len(dates), np.nan)

        mapped = series.copy()
        mapped.index = pd.to_datetime(mapped.index).normalize()

        if mapped.index.has_duplicates:
            mapped = mapped.groupby(mapped.index).mean()

        return dates.map(mapped).to_numpy()

    def _add_load_features(self, out: pd.DataFrame, frame: pd.DataFrame) -> None:
        actual = frame["actual_mw"]
        hour = out["hour"]

        lag_24 = self._same_hour_shift(actual, 1)
        lag_48 = self._same_hour_shift(actual, 2)
        lag_72 = self._same_hour_shift(actual, 3)
        lag_168 = self._same_hour_shift(actual, 7)
        lag_336 = self._same_hour_shift(actual, 14)

        out["actual_lag_24h"] = lag_24.where(hour < 12)
        out["actual_lag_48h"] = lag_48
        out["actual_lag_72h"] = lag_72
        out["actual_lag_168h"] = lag_168
        out["actual_lag_336h"] = lag_336
        out["actual_latest_same_hour_known"] = lag_24.where(hour < 12, lag_48)

        for window in (7, 14, 28):
            out[f"actual_roll_mean_{window}"] = self._rolling_by_hour(
                actual,
                window,
                "mean",
            )

        out["actual_roll_std_14"] = self._rolling_by_hour(actual, 14, "std")
        out["actual_trend_7_28"] = (
            out["actual_roll_mean_7"] - out["actual_roll_mean_28"]
        )

        target_date = out.index.normalize()
        previous_date = target_date - pd.DateOffset(days=1)
        previous_two_date = target_date - pd.DateOffset(days=2)

        morning_actual = actual[frame.index.hour < 12]
        morning_mean = morning_actual.groupby(
            morning_actual.index.normalize()
        ).mean()

        daily_total = actual.groupby(actual.index.normalize()).agg(
            lambda group: group.sum(min_count=1)
        )

        out["prev_morning_load_mean"] = self._map_date_series(
            morning_mean,
            previous_date,
        )
        out["prev2_morning_load_mean"] = self._map_date_series(
            morning_mean,
            previous_two_date,
        )
        out["morning_load_diff"] = (
            out["prev_morning_load_mean"] - out["prev2_morning_load_mean"]
        )
        out["prev2_daily_total"] = self._map_date_series(
            daily_total,
            previous_two_date,
        )

    def _add_tso_features(self, out: pd.DataFrame, frame: pd.DataFrame) -> None:
        tso = frame["tso_mw"]

        lag_24 = self._same_hour_shift(tso, 1)
        lag_48 = self._same_hour_shift(tso, 2)
        lag_168 = self._same_hour_shift(tso, 7)

        out["tso_lag_24h"] = lag_24
        out["tso_lag_48h"] = lag_48
        out["tso_lag_168h"] = lag_168
        out["tso_latest_same_hour_known"] = lag_24.fillna(lag_48)
        out["tso_diff_24h"] = out["tso_mw"] - out["tso_lag_24h"]

        out["tso_daily_total"] = tso.groupby(tso.index.normalize()).transform(
            lambda group: group.sum(min_count=1)
        )
        
        out["tso_daily_max"] = tso.groupby(tso.index.normalize()).transform("max")
        out["tso_daily_min"] = tso.groupby(tso.index.normalize()).transform("min")
        out["tso_daily_range"] = out["tso_daily_max"] - out["tso_daily_min"]
        
        safe_max = out["tso_daily_max"].replace(0, 1)
        out["tso_pct_of_daily_max"] = out["tso_mw"] / safe_max

        tso_morning = tso[frame.index.hour < 12]
        tso_morning_mean = tso_morning.groupby(
            tso_morning.index.normalize()
        ).mean()

        previous_date = out.index.normalize() - pd.DateOffset(days=1)
        out["prev_tso_morning_mean"] = self._map_date_series(
            tso_morning_mean,
            previous_date,
        )

    def _add_error_features(self, out: pd.DataFrame, frame: pd.DataFrame) -> None:
        error = frame["actual_mw"] - frame["tso_mw"]

        out["error_lag_48h"] = self._same_hour_shift(error, 2)
        out["error_lag_168h"] = self._same_hour_shift(error, 7)

        for window in (7, 14):
            out[f"error_roll_mean_{window}"] = self._rolling_by_hour(
                error,
                window,
                "mean",
            )

        out["error_roll_std_14"] = self._rolling_by_hour(error, 14, "std")

    def _add_weather_features(
        self,
        out: pd.DataFrame,
        weather: pd.DataFrame | None,
    ) -> None:
        if weather is None:
            raise ValueError("weather dataframe is required when include_weather is true")

        aligned = weather.reindex(out.index)
        required = {
            "temperature_2m",
            "apparent_temperature",
            "cloud_cover",
            "shortwave_radiation",
            "wind_speed_10m",
            "precipitation",
        }

        missing = required.difference(aligned.columns)
        if missing:
            raise KeyError(f"missing weather columns {sorted(missing)}")

        for column in sorted(required):
            out[f"weather_{column}"] = aligned[column]

        out["weather_heating_degree"] = (
            18.0 - out["weather_temperature_2m"]
        ).clip(lower=0.0)
        out["weather_cooling_degree"] = (
            out["weather_temperature_2m"] - 22.0
        ).clip(lower=0.0)
        
        for window in (6, 12, 24):
            temp_roll = out["weather_temperature_2m"].rolling(window, min_periods=1).mean()
            rad_roll = out["weather_shortwave_radiation"].rolling(window, min_periods=1).mean()
            
            out[f"weather_temp_roll_{window}"] = temp_roll
            out[f"weather_radiation_roll_{window}"] = rad_roll
            
            out[f"weather_heating_degree_roll_{window}"] = (18.0 - temp_roll).clip(lower=0.0)
            out[f"weather_cooling_degree_roll_{window}"] = (temp_roll - 22.0).clip(lower=0.0)