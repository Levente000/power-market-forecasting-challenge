from __future__ import annotations

import pandas as pd

from src.config import Settings


class LoadRepository:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def load_hourly(self) -> pd.DataFrame:
        settings = self._settings

        frame = pd.read_csv(settings.data_path)

        if settings.timestamp_column not in frame.columns:
            raise KeyError(f"missing timestamp column {settings.timestamp_column}")

        timestamp = pd.to_datetime(
            frame[settings.timestamp_column],
            format=settings.timestamp_format,
            utc=True,
        )

        frame = frame.assign(timestamp=timestamp).set_index("timestamp")
        frame.index = frame.index.tz_convert(settings.timezone)

        columns = [settings.actual_column, settings.tso_column]

        for column in columns:
            if column not in frame.columns:
                raise KeyError(f"missing data column {column}")
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

        hourly_mean = frame[columns].resample("1h").mean()
        hourly_mean.columns = ["actual_mw", "tso_mw"]
        
        hourly_std = frame[columns].resample("1h").std()
        hourly_std.columns = ["actual_std", "tso_std"]
        
        hourly_max = frame[columns].resample("1h").max()
        hourly_min = frame[columns].resample("1h").min()
        hourly_range = hourly_max - hourly_min
        hourly_range.columns = ["actual_range", "tso_range"]

        hourly = pd.concat([hourly_mean, hourly_std, hourly_range], axis=1)
        hourly = hourly.sort_index()

        return hourly