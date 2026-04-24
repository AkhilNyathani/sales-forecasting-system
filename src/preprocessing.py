from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .utils import get_logger, infer_frequency

LOGGER = get_logger("preprocessing")


def load_sales_data(source: str | Path | Any) -> pd.DataFrame:
    try:
        dataframe = pd.read_csv(source)
    except Exception as exc:
        raise ValueError(f"Unable to read the CSV file: {exc}") from exc

    if dataframe.empty:
        raise ValueError("The provided CSV file is empty.")

    return dataframe


def preprocess_sales_data(
    dataframe: pd.DataFrame,
    date_column: str = "date",
    target_column: str = "sales",
) -> tuple[pd.DataFrame, str]:
    required_columns = {date_column, target_column}
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing_display = ", ".join(sorted(missing_columns))
        raise ValueError(f"Missing required columns: {missing_display}")

    cleaned_df = dataframe[[date_column, target_column]].copy()
    cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors="coerce")
    cleaned_df[target_column] = pd.to_numeric(cleaned_df[target_column], errors="coerce")

    invalid_date_count = int(cleaned_df[date_column].isna().sum())
    if invalid_date_count:
        LOGGER.warning("Dropped %s rows because the date could not be parsed.", invalid_date_count)

    cleaned_df = cleaned_df.dropna(subset=[date_column])
    if cleaned_df.empty:
        raise ValueError("No valid date values were found in the dataset.")

    cleaned_df = (
        cleaned_df.groupby(date_column, as_index=False)
        .agg({target_column: lambda values: values.sum(min_count=1)})
        .sort_values(date_column)
        .reset_index(drop=True)
    )

    frequency = infer_frequency(pd.DatetimeIndex(cleaned_df[date_column]))
    full_date_index = pd.date_range(
        start=cleaned_df[date_column].min(),
        end=cleaned_df[date_column].max(),
        freq=frequency,
    )

    cleaned_df = cleaned_df.set_index(date_column).reindex(full_date_index)
    cleaned_df.index.name = date_column
    cleaned_df[target_column] = (
        cleaned_df[target_column]
        .interpolate(method="time")
        .ffill()
        .bfill()
        .astype(float)
        .round(2)
    )

    if cleaned_df[target_column].isna().any():
        raise ValueError("Missing sales values remain after preprocessing.")

    cleaned_df = cleaned_df.reset_index().rename(columns={"index": date_column})

    if len(cleaned_df) < 30:
        raise ValueError("At least 30 cleaned observations are required for forecasting.")

    return cleaned_df, frequency
