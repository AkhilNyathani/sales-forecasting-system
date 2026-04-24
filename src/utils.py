from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

LOGGER_NAME = "sales_forecasting"


class FallbackLinearRegressor:
    """Small numpy-based fallback used when scikit-learn is unavailable."""

    def __init__(self) -> None:
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    def fit(self, features: pd.DataFrame | np.ndarray, target: pd.Series | np.ndarray) -> "FallbackLinearRegressor":
        x_values = np.asarray(features, dtype=float)
        y_values = np.asarray(target, dtype=float)
        design_matrix = np.column_stack([np.ones(len(x_values)), x_values])
        solution, *_ = np.linalg.lstsq(design_matrix, y_values, rcond=None)
        self.intercept_ = float(solution[0])
        self.coef_ = solution[1:]
        return self

    def predict(self, features: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise ValueError("The fallback linear model must be fitted before prediction.")
        x_values = np.asarray(features, dtype=float)
        return np.dot(x_values, self.coef_) + self.intercept_


@dataclass
class ForecastArtifacts:
    model_name: str
    model: Any
    feature_columns: list[str]
    feature_config: dict[str, list[int]]
    frequency: str
    metrics: dict[str, float]
    residual_std: float
    training_rows: int


def configure_logging(level: int = logging.INFO) -> None:
    logger = logging.getLogger(LOGGER_NAME)
    if logger.handlers:
        return
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def infer_frequency(date_index: pd.DatetimeIndex) -> str:
    if len(date_index) < 3:
        return "D"

    inferred_frequency = pd.infer_freq(date_index)
    if inferred_frequency:
        return inferred_frequency

    deltas = date_index.to_series().diff().dropna()
    if deltas.empty:
        return "D"

    return to_offset(deltas.mode().iloc[0]).freqstr


def resolve_model_path(model_path: str | Path) -> Path:
    resolved = Path(model_path).resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def save_model(artifacts: ForecastArtifacts, model_path: str | Path) -> Path:
    resolved_path = resolve_model_path(model_path)
    with resolved_path.open("wb") as file_pointer:
        pickle.dump(artifacts, file_pointer)
    return resolved_path


def load_model(model_path: str | Path) -> ForecastArtifacts:
    resolved_path = Path(model_path).resolve()
    with resolved_path.open("rb") as file_pointer:
        return pickle.load(file_pointer)


def determine_feature_config(series_length: int) -> dict[str, list[int]]:
    if series_length < 21:
        raise ValueError("At least 21 observations are required to build forecasting features.")

    lags = [1]
    for candidate in (7, 14):
        if series_length > candidate + 5:
            lags.append(candidate)

    rolling_windows = [window for window in (7, 14) if series_length > window + 5]
    return {"lags": lags, "rolling_windows": rolling_windows}


def get_feature_columns(feature_config: dict[str, list[int]]) -> list[str]:
    base_features = [
        "time_idx",
        "month",
        "quarter",
        "day_of_week",
        "day_of_month",
        "week_of_year",
        "is_month_start",
        "is_month_end",
    ]
    lag_features = [f"lag_{lag}" for lag in feature_config["lags"]]
    rolling_features: list[str] = []
    for window in feature_config["rolling_windows"]:
        rolling_features.append(f"rolling_mean_{window}")
        rolling_features.append(f"rolling_std_{window}")
    return base_features + lag_features + rolling_features


def build_supervised_frame(
    dataframe: pd.DataFrame,
    feature_config: dict[str, list[int]],
    target_column: str = "sales",
) -> pd.DataFrame:
    frame = dataframe.copy()
    frame["time_idx"] = np.arange(len(frame))
    frame["month"] = frame["date"].dt.month
    frame["quarter"] = frame["date"].dt.quarter
    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["day_of_month"] = frame["date"].dt.day
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["is_month_start"] = frame["date"].dt.is_month_start.astype(int)
    frame["is_month_end"] = frame["date"].dt.is_month_end.astype(int)

    for lag in feature_config["lags"]:
        frame[f"lag_{lag}"] = frame[target_column].shift(lag)

    for window in feature_config["rolling_windows"]:
        shifted_series = frame[target_column].shift(1)
        frame[f"rolling_mean_{window}"] = shifted_series.rolling(window).mean()
        frame[f"rolling_std_{window}"] = shifted_series.rolling(window).std(ddof=0)

    frame = frame.dropna().reset_index(drop=True)
    if frame.empty:
        raise ValueError("Unable to create training features from the provided data.")
    return frame


def make_recursive_feature_row(
    history_values: list[float],
    next_date: pd.Timestamp,
    time_idx: int,
    feature_config: dict[str, list[int]],
) -> pd.DataFrame:
    history_series = pd.Series(history_values, dtype=float)
    feature_row: dict[str, float | int] = {
        "time_idx": time_idx,
        "month": next_date.month,
        "quarter": next_date.quarter,
        "day_of_week": next_date.dayofweek,
        "day_of_month": next_date.day,
        "week_of_year": int(next_date.isocalendar().week),
        "is_month_start": int(next_date.is_month_start),
        "is_month_end": int(next_date.is_month_end),
    }

    for lag in feature_config["lags"]:
        feature_row[f"lag_{lag}"] = float(history_series.iloc[-lag])

    for window in feature_config["rolling_windows"]:
        rolling_slice = history_series.iloc[-window:]
        feature_row[f"rolling_mean_{window}"] = float(rolling_slice.mean())
        feature_row[f"rolling_std_{window}"] = (
            float(rolling_slice.std(ddof=0)) if len(rolling_slice) > 1 else 0.0
        )

    ordered_columns = get_feature_columns(feature_config)
    return pd.DataFrame([feature_row], columns=ordered_columns)


def create_future_dates(last_date: pd.Timestamp, horizon: int, frequency: str) -> pd.DatetimeIndex:
    offset = to_offset(frequency)
    start_date = last_date + offset
    return pd.date_range(start=start_date, periods=horizon, freq=offset)


def recursive_forecast(
    model: Any,
    history_df: pd.DataFrame,
    horizon: int,
    feature_config: dict[str, list[int]],
    frequency: str,
    target_column: str = "sales",
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("Forecast horizon must be at least 1.")

    history_values = history_df[target_column].astype(float).tolist()
    last_date = pd.to_datetime(history_df["date"].iloc[-1])
    future_dates = create_future_dates(last_date, horizon, frequency)

    forecast_records: list[dict[str, float | int | pd.Timestamp]] = []
    for step, future_date in enumerate(future_dates, start=1):
        feature_row = make_recursive_feature_row(
            history_values=history_values,
            next_date=future_date,
            time_idx=len(history_values),
            feature_config=feature_config,
        )
        predicted_value = float(model.predict(feature_row)[0])
        predicted_value = max(predicted_value, 0.0)
        history_values.append(predicted_value)
        forecast_records.append(
            {
                "date": future_date,
                "prediction": predicted_value,
                "step": step,
            }
        )

    return pd.DataFrame(forecast_records)


def calculate_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    mae = float(np.mean(np.abs(actual_values - predicted_values)))
    rmse = float(np.sqrt(np.mean(np.square(actual_values - predicted_values))))
    return {"mae": mae, "rmse": rmse}


def build_prediction_intervals(
    predictions: pd.Series | np.ndarray,
    residual_std: float,
    z_score: float = 1.96,
) -> tuple[np.ndarray, np.ndarray]:
    prediction_values = np.asarray(predictions, dtype=float)
    base_std = residual_std
    if not np.isfinite(base_std) or base_std <= 0:
        base_std = max(float(np.std(prediction_values) * 0.1), 1.0)

    horizon_scaler = np.sqrt(np.arange(1, len(prediction_values) + 1))
    spread = z_score * base_std * horizon_scaler
    lower_bound = np.maximum(prediction_values - spread, 0.0)
    upper_bound = prediction_values + spread
    return lower_bound, upper_bound
