from __future__ import annotations

import pandas as pd

from src.config import Settings
from src.data.weather_repository import WeatherRepository


def load_weather(settings: Settings, hourly: pd.DataFrame) -> pd.DataFrame | None:
    if not settings.include_weather:
        return None

    if settings.weather_cache_path.exists():
        weather = pd.read_csv(
            settings.weather_cache_path,
            index_col=0,
            parse_dates=True,
        )

        weather.index = pd.to_datetime(weather.index)

        if weather.index.tz is None:
            weather.index = weather.index.tz_localize("UTC")
        else:
            weather.index = weather.index.tz_convert("UTC")

        weather = weather.sort_index()
        return weather.tz_convert(settings.timezone)

    start_date = hourly.index.min().strftime("%Y-%m-%d")
    end_date = hourly.index.max().strftime("%Y-%m-%d")

    repository = WeatherRepository(settings)
    weather = repository.fetch_historical_forecasts(start_date, end_date)

    settings.weather_cache_path.parent.mkdir(parents=True, exist_ok=True)
    weather.to_csv(settings.weather_cache_path)

    return weather.tz_convert(settings.timezone)