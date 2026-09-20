from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error

from src.config import Objective, Settings, TargetMode
from src.domain.schemas import TuningResult
from src.models.lightgbm_model import LightGbmForecaster


class OptunaTuner:
    def __init__(self, settings: Settings, feature_names: list[str]) -> None:
        self._settings = settings
        self._feature_names = feature_names

    def tune(self, dataset: pd.DataFrame, n_trials: int = 50) -> TuningResult:
        folds = self._build_folds(dataset)

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self._settings.random_seed),
        )

        study.optimize(
            lambda trial: self._objective(trial, dataset, folds),
            n_trials=n_trials,
        )

        best_params = study.best_params
        best_iteration = self._fit_final_and_get_iteration(dataset, best_params)

        return TuningResult(
            best_validation_mae=float(study.best_value),
            best_params=best_params,
            best_iteration=best_iteration,
        )

    def build_final_forecaster(
        self,
        dataset: pd.DataFrame,
        result: TuningResult,
    ) -> LightGbmForecaster:
        target_column = self._target_column()
        unique_dates = self._valid_dates(dataset)

        if self._settings.holdout_days > 0:
            tuning_dates = unique_dates[: -self._settings.holdout_days]
        else:
            tuning_dates = unique_dates

        if len(tuning_dates) == 0:
            raise ValueError("not enough data for final training")

        train_mask = dataset.index.normalize().isin(set(tuning_dates))
        train = dataset[train_mask].dropna(subset=[target_column])

        if train.empty:
            raise ValueError("training frame is empty")

        forecaster = LightGbmForecaster.create(
            settings=self._settings,
            feature_names=self._feature_names,
            params=result.best_params,
            best_iteration=result.best_iteration,
        )

        forecaster.fit(train[self._feature_names], train[target_column])

        return forecaster

    def _target_column(self) -> str:
        if self._settings.target_mode == TargetMode.RESIDUAL:
            return "residual_mw"
        return "actual_mw"

    def _valid_dates(self, dataset: pd.DataFrame) -> np.ndarray:
        target_column = self._target_column()
        valid = dataset.dropna(subset=[target_column])
        return np.sort(pd.Series(valid.index.normalize()).unique())

    def _build_folds(
        self,
        dataset: pd.DataFrame,
    ) -> list[tuple[set[Any], set[Any], set[Any]]]:
        unique_dates = self._valid_dates(dataset)

        if self._settings.holdout_days > 0:
            tuning_dates = unique_dates[: -self._settings.holdout_days]
        else:
            tuning_dates = unique_dates

        validation_days = 14
        early_stopping_days = 7
        fold_count = 3
        folds = []

        for fold_index in range(fold_count):
            validation_end = len(tuning_dates) - fold_index * validation_days
            validation_start = validation_end - validation_days
            early_start = validation_start - early_stopping_days
            train_end = early_start

            if train_end <= 30:
                break

            train_dates = set(tuning_dates[:train_end])
            early_dates = set(tuning_dates[early_start:validation_start])
            validation_dates = set(tuning_dates[validation_start:validation_end])

            folds.append((train_dates, early_dates, validation_dates))

        if not folds:
            raise ValueError("not enough history to build validation folds")

        return folds

    def _date_mask(self, dataset: pd.DataFrame, dates: set[Any]) -> pd.Series:
        return dataset.index.normalize().isin(dates)

    def _suggest_params(self, trial: optuna.trial.Trial) -> dict[str, Any]:
        return {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 1, 7),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }

    def _make_model(self, params: dict[str, Any], n_estimators: int) -> lgb.LGBMRegressor:
        objective = (
            "regression_l1"
            if self._settings.objective == Objective.L1
            else "quantile"
        )

        model = lgb.LGBMRegressor(
            objective=objective,
            n_estimators=n_estimators,
            random_state=self._settings.random_seed,
            n_jobs=self._settings.n_jobs,
            verbosity=-1,
            **params,
        )

        if objective == "quantile":
            model.set_params(alpha=self._settings.quantile_tau)

        return model

    def _to_actual_predictions(
        self,
        frame: pd.DataFrame,
        predictions: np.ndarray,
    ) -> np.ndarray:
        if self._settings.target_mode == TargetMode.RESIDUAL:
            return frame["tso_mw"].to_numpy(dtype="float64") + predictions
        return predictions

    def _objective(
        self,
        trial: optuna.trial.Trial,
        dataset: pd.DataFrame,
        folds: list[tuple[set[Any], set[Any], set[Any]]],
    ) -> float:
        params = self._suggest_params(trial)
        target_column = self._target_column()
        scores = []

        for train_dates, early_dates, validation_dates in folds:
            train_mask = self._date_mask(dataset, train_dates)
            early_mask = self._date_mask(dataset, early_dates)
            validation_mask = self._date_mask(dataset, validation_dates)

            train = dataset[train_mask].dropna(subset=[target_column])
            early = dataset[early_mask].dropna(subset=[target_column])
            validation = dataset[validation_mask].dropna(subset=[target_column])

            if train.empty or validation.empty:
                continue

            model = self._make_model(params, n_estimators=2000)

            train_x = train[self._feature_names]
            train_y = train[target_column]

            fit_arguments = {}

            if not early.empty:
                fit_arguments["eval_X"] = early[self._feature_names]
                fit_arguments["eval_y"] = early[target_column]
                fit_arguments["callbacks"] = [
                    lgb.early_stopping(stopping_rounds=50, verbose=False)
                ]

            model.fit(
                train_x,
                train_y,
                **fit_arguments,
            )

            predictions = model.predict(validation[self._feature_names]).astype("float64")
            predicted_actual = self._to_actual_predictions(validation, predictions)

            actual = validation["actual_mw"].to_numpy(dtype="float64")
            valid = np.isfinite(actual) & np.isfinite(predicted_actual)

            if valid.sum() == 0:
                continue

            score = mean_absolute_error(actual[valid], predicted_actual[valid])
            scores.append(float(score))

        if not scores:
            return 1_000_000.0

        return float(np.mean(scores))

    def _fit_final_and_get_iteration(
        self,
        dataset: pd.DataFrame,
        params: dict[str, Any],
    ) -> int:
        unique_dates = self._valid_dates(dataset)

        if self._settings.holdout_days > 0:
            tuning_dates = unique_dates[: -self._settings.holdout_days]
        else:
            tuning_dates = unique_dates

        if len(tuning_dates) <= 7:
            raise ValueError("not enough tuning dates")

        train_dates = set(tuning_dates[:-7])
        evaluation_dates = set(tuning_dates[-7:])
        target_column = self._target_column()

        train_mask = self._date_mask(dataset, train_dates)
        evaluation_mask = self._date_mask(dataset, evaluation_dates)

        train = dataset[train_mask].dropna(subset=[target_column])
        evaluation = dataset[evaluation_mask].dropna(subset=[target_column])

        model = self._make_model(params, n_estimators=2000)

        fit_arguments = {}

        if not evaluation.empty:
            fit_arguments["eval_X"] = evaluation[self._feature_names]
            fit_arguments["eval_y"] = evaluation[target_column]
            fit_arguments["callbacks"] = [
                lgb.early_stopping(stopping_rounds=50, verbose=False)
            ]

        model.fit(
            train[self._feature_names],
            train[target_column],
            **fit_arguments,
        )

        best_iteration = getattr(model, "best_iteration_", None)

        if best_iteration is None or best_iteration <= 0:
            return 800

        return int(best_iteration)