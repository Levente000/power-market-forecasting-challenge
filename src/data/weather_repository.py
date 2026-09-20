from __future__ import annotations

import pandas as pd
import requests

from src.config import Settings


class WeatherRepository:
    CITIES = [
        {"lat": 47.4979, "lon": 19.0402},
        {"lat": 47.5316, "lon": 21.6273},
        {"lat": 46.2530, "lon": 20.1414},
        {"lat": 48.1034, "lon": 20.7784},
        {"lat": 46.0767, "lon": 18.2283},
        {"lat": 47.6875, "lon": 17.6504},
    ]

    VARIABLES = [
        "temperature_2m",
        "apparent_temperature",
        "cloud_cover",
        "shortwave_radiation",
        "wind_speed_10m",
        "precipitation",
    ]

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def fetch_historical_forecasts(self, start_date: str, end_date: str) -> pd.DataFrame:
        lats = ",".join(str(city["lat"]) for city in self.CITIES)
        lons = ",".join(str(city["lon"]) for city in self.CITIES)
        variables = ",".join(self.VARIABLES)

        url = "https://historical-forecast-api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lats,
            "longitude": lons,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": variables,
            "timezone": "UTC",
        }

        response = requests.get(url, params=params, timeout=180)
        response.raise_for_status()
        payload = response.json()

        locations = payload if isinstance(payload, list) else [payload]
        frames = []

        for location_payload in locations:
            hourly = location_payload["hourly"]
            frame = pd.DataFrame({"time": pd.to_datetime(hourly["time"])})

            for variable in self.VARIABLES:
                frame[variable] = hourly[variable]

            frame = frame.set_index("time")
            frames.append(frame)

        weather = pd.concat(frames).groupby(level=0).mean()
        weather = weather.sort_index()

        if weather.index.tz is None:
            weather.index = weather.index.tz_localize("UTC")
        else:
            weather.index = weather.index.tz_convert("UTC")

        return weather