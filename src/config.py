from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, field_validator


class TargetMode(str, Enum):
    RESIDUAL = "residual"
    ABSOLUTE = "absolute"


class Objective(str, Enum):
    L1 = "l1"
    QUANTILE = "quantile"


class Settings(BaseModel):
    data_path: Path
    artifact_path: Path
    output_path: Path
    weather_cache_path: Path = Path("data/weather_cache.csv")
    timezone: str = "Europe/Budapest"
    timestamp_column: str
    timestamp_format: str
    actual_column: str
    tso_column: str
    holdout_days: int = Field(default=28, ge=0)
    include_weather: bool = False
    target_mode: TargetMode = TargetMode.RESIDUAL
    objective: Objective = Objective.L1
    quantile_tau: float = Field(default=0.5, ge=0.0, le=1.0)
    random_seed: int = 42
    n_jobs: int = -1

    @field_validator(
        "data_path",
        "artifact_path",
        "output_path",
        "weather_cache_path",
        mode="before",
    )
    @classmethod
    def _to_path(cls, value: Any) -> Path:
        return Path(value)


def load_settings(path: str | Path) -> Settings:
    raw = Path(path).read_text(encoding="utf-8")
    return Settings.model_validate(json.loads(raw))