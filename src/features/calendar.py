from __future__ import annotations

import holidays
import numpy as np
import pandas as pd


def _holiday_dates(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    if len(index) == 0:
        return set()

    year_list = list(range(index.min().year, index.max().year + 1))
    calendar = holidays.country_holidays("HU", years=year_list)

    dates = {pd.Timestamp(day).normalize() for day in calendar.keys()}

    for year in year_list:
        dates.add(pd.Timestamp(year=year, month=12, day=24))
        dates.add(pd.Timestamp(year=year, month=12, day=31))

    return dates


def add_calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    out = pd.DataFrame(index=index)
    holiday_dates = _holiday_dates(index)
    normalized = index.normalize()

    out["hour"] = index.hour.astype("int64")
    out["day_of_week"] = index.dayofweek.astype("int64")
    out["month"] = index.month.astype("int64")
    out["day_of_year"] = index.dayofyear.astype("int64")
    out["week_of_year"] = index.isocalendar().week.astype("int64").to_numpy()
    out["is_weekend"] = (index.dayofweek >= 5).astype("int64")
    out["is_holiday"] = normalized.isin(holiday_dates).astype("int64")

    is_dst = index.to_series().apply(lambda x: bool(x.dst())).astype("int64")
    out["is_dst"] = is_dst.to_numpy()

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)

    out["day_of_week_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["day_of_week_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

    out["day_of_year_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
    out["day_of_year_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)

    out["horizon_hours"] = (12 + out["hour"]).astype("int64")

    return out