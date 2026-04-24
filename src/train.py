from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import (
    FallbackLinearRegressor,
    ForecastArtifacts,
    build_supervised_frame,
    calculate_metrics,
    determine_feature_config,
    get_feature_columns,
    get_logger,
    infer_frequency,
    recursive_forecast,
    save_model,
)

try:
    from sklearn.linear_model import LinearRegression
except ImportError:  # pragma: no cover - environment-dependent fallback
    LinearRegression = None

try:
    from xgboost import XGBRegressor
except ImportError:  # pragma: no cover - environment-dependent fallback
    XGBRegressor = None

LOGGER = get_logger("train")


def get_available_models() -> list[str]:
    models = ["Linear Regression"]
    if XGBRegressor is not None:
        models.append("XGBoost")
    return models


def _choose_validation_window(total_rows: int) -> int:
    proposed_window = max(7, min(30, total_rows // 5))
    return min(proposed_window, total_rows - 21)


def _initialize_model(model_name: str):
    if model_name == "Linear Regression":
        if LinearRegression is not None:
            return LinearRegression()
        LOGGER.warning(
            "scikit-learn is unavailable in the current environment. Falling back to a numpy-based "
            "linear regression implementation."
        )
        return FallbackLinearRegressor()

    if model_name == "XGBoost":
        if XGBRegressor is None:
            raise ImportError(
                "XGBoost is not installed. Install dependencies from requirements.txt to use this model."
            )
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=1,
        )

    raise ValueError(f"Unsupported model selection: {model_name}")


def train_forecasting_model(
    clean_df: pd.DataFrame,
    model_name: str,
    model_path: str | Path,
    target_column: str = "sales",
) -> ForecastArtifacts:
    if clean_df.empty:
        raise ValueError("The cleaned dataframe is empty.")

    validation_window = _choose_validation_window(len(clean_df))
    if validation_window < 7:
        raise ValueError("The dataset is too short to create a validation window.")

    train_df = clean_df.iloc[:-validation_window].reset_index(drop=True)
    validation_df = clean_df.iloc[-validation_window:].reset_index(drop=True)
    feature_config = determine_feature_config(len(train_df))
    feature_columns = get_feature_columns(feature_config)
    frequency = infer_frequency(pd.DatetimeIndex(clean_df["date"]))

    training_frame = build_supervised_frame(
        dataframe=train_df,
        feature_config=feature_config,
        target_column=target_column,
    )

    validation_model = _initialize_model(model_name)
    validation_model.fit(training_frame[feature_columns], training_frame[target_column])

    validation_forecast = recursive_forecast(
        model=validation_model,
        history_df=train_df,
        horizon=validation_window,
        feature_config=feature_config,
        frequency=frequency,
        target_column=target_column,
    )

    metrics = calculate_metrics(
        actual=validation_df[target_column],
        predicted=validation_forecast["prediction"],
    )
    residuals = validation_df[target_column].to_numpy() - validation_forecast["prediction"].to_numpy()
    residual_std = float(pd.Series(residuals).std(ddof=1))

    full_training_frame = build_supervised_frame(
        dataframe=clean_df,
        feature_config=feature_config,
        target_column=target_column,
    )

    final_model = _initialize_model(model_name)
    final_model.fit(full_training_frame[feature_columns], full_training_frame[target_column])

    artifacts = ForecastArtifacts(
        model_name=model_name,
        model=final_model,
        feature_columns=feature_columns,
        feature_config=feature_config,
        frequency=frequency,
        metrics=metrics,
        residual_std=residual_std,
        training_rows=len(clean_df),
    )
    save_model(artifacts, model_path)

    LOGGER.info(
        "Model trained successfully with %s. Validation MAE: %.3f | Validation RMSE: %.3f",
        model_name,
        metrics["mae"],
        metrics["rmse"],
    )
    return artifacts
