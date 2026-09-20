from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.domain.schemas import ForecastRow


def write_forecast_csv(rows: list[ForecastRow], path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)

    frame = pd.DataFrame([row.model_dump(mode="json") for row in rows])
    frame.to_csv(target, index=False)