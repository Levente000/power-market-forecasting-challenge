from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import pandas as pd

from src.config import Objective, Settings, TargetMode
from src.domain.schemas import ModelMetadata


class LightGbmForecaster:
    def __init__(self, metadata: ModelMetadata, model: lgb.LGBMRegressor) -> None:
        self._metadata = metadata
        self._model = model

    @property
    def metadata(self) -> ModelMetadata:
        return self._metadata

    @property
    def feature_names(self) -> list[str]:
        return self._metadata.feature_names

    @classmethod
    def create(
        cls,
        settings: Settings,
        feature_names: list[str],
        params: dict[str, Any],
        best_iteration: int | None = None,
    ) -> "LightGbmForecaster":
        model_params = params.copy()

        if best_iteration is not None:
            model_params["n_estimators"] = best_iteration
        else:
            model_params.setdefault("n_estimators", 800)

        objective = (
            "regression_l1"
            if settings.objective == Objective.L1
            else "quantile"
        )

        model = lgb.LGBMRegressor(
            objective=objective,
            random_state=settings.random_seed,
            n_jobs=settings.n_jobs,
            verbosity=-1,
            **model_params,
        )

        if settings.objective == Objective.QUANTILE:
            model.set_params(alpha=settings.quantile_tau)

        metadata = ModelMetadata(
            feature_names=feature_names,
            target_mode=settings.target_mode,
            objective=settings.objective,
            quantile_tau=settings.quantile_tau,
            params=model_params,
            created_at=datetime.now(timezone.utc),
        )

        return cls(metadata=metadata, model=model)

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: list[tuple[pd.DataFrame, pd.Series]] | None = None,
        early_stopping_rounds: int = 50,
    ) -> "LightGbmForecaster":
        train_frame = X[self._metadata.feature_names]
        
        eval_X = None
        eval_y = None
        callbacks = None

        if eval_set is not None and len(eval_set) > 0:
            eval_X = eval_set[0][0][self._metadata.feature_names]
            eval_y = eval_set[0][1]
            callbacks = [
                lgb.early_stopping(
                    stopping_rounds=early_stopping_rounds,
                    verbose=False,
                )
            ]

        self._model.fit(
            train_frame,
            y,
            eval_X=eval_X,
            eval_y=eval_y,
            callbacks=callbacks,
        )

        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        frame = X[self._metadata.feature_names]
        values = self._model.predict(frame)
        return pd.Series(values, index=X.index)

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "metadata": self._metadata.model_dump(),
            "model": self._model,
        }

        joblib.dump(payload, target)

    @classmethod
    def load(cls, path: str | Path) -> "LightGbmForecaster":
        payload = joblib.load(Path(path))
        metadata = ModelMetadata.model_validate(payload["metadata"])
        return cls(metadata=metadata, model=payload["model"])