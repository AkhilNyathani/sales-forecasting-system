from __future__ import annotations

import pandas as pd

from .utils import ForecastArtifacts, build_prediction_intervals, recursive_forecast


def generate_forecast(
    history_df: pd.DataFrame,
    artifacts: ForecastArtifacts,
    horizon: int,
    target_column: str = "sales",
) -> pd.DataFrame:
    forecast_df = recursive_forecast(
        model=artifacts.model,
        history_df=history_df,
        horizon=horizon,
        feature_config=artifacts.feature_config,
        frequency=artifacts.frequency,
        target_column=target_column,
    )

    lower_bound, upper_bound = build_prediction_intervals(
        predictions=forecast_df["prediction"],
        residual_std=artifacts.residual_std,
    )
    forecast_df["forecast"] = forecast_df["prediction"].round(2)
    forecast_df["lower_bound"] = lower_bound.round(2)
    forecast_df["upper_bound"] = upper_bound.round(2)
    return forecast_df[["date", "forecast", "lower_bound", "upper_bound", "step"]]
