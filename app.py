from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import generate_forecast
from src.preprocessing import load_sales_data, preprocess_sales_data
from src.train import get_available_models, train_forecasting_model
from src.utils import configure_logging, get_logger

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "sample_sales.csv"
MODEL_PATH = BASE_DIR / "models" / "forecasting_model.pkl"

configure_logging()
LOGGER = get_logger("app")


def create_forecast_figure(
    history_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> go.Figure:
    figure = go.Figure()

    figure.add_trace(
        go.Scatter(
            x=history_df["date"],
            y=history_df["sales"],
            mode="lines+markers",
            name="Historical Sales",
            line={"color": "#1f77b4", "width": 2},
            marker={"size": 5},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["upper_bound"],
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["lower_bound"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(255, 127, 14, 0.18)",
            line={"width": 0},
            hoverinfo="skip",
            name="95% Prediction Interval",
        )
    )
    figure.add_trace(
        go.Scatter(
            x=forecast_df["date"],
            y=forecast_df["forecast"],
            mode="lines+markers",
            name="Forecast",
            line={"color": "#ff7f0e", "width": 3},
            marker={"size": 6},
        )
    )

    figure.update_layout(
        title="Historical Sales and Forecast",
        xaxis_title="Date",
        yaxis_title="Sales",
        template="plotly_white",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 70, "b": 20},
        hovermode="x unified",
    )
    return figure


def main() -> None:
    st.set_page_config(
        page_title="Sales Forecasting System",
        layout="wide",
    )

    st.title("Sales Forecasting System")
    st.caption(
        "Upload a CSV with `date` and `sales` columns, or use the bundled sample dataset "
        "to run the app immediately."
    )

    uploaded_file = st.file_uploader(
        "Upload historical sales data",
        type=["csv"],
        help="The CSV must include `date` and `sales` columns.",
    )

    try:
        raw_df = load_sales_data(uploaded_file or DATA_PATH)
        clean_df, frequency = preprocess_sales_data(raw_df)
    except Exception as exc:  # pragma: no cover - UI handling
        LOGGER.exception("Failed to load or preprocess sales data.")
        st.error(f"Unable to prepare the dataset: {exc}")
        st.stop()

    if uploaded_file is None:
        st.info("Using the bundled sample dataset. Upload your own CSV at any time.")

    summary_col_1, summary_col_2, summary_col_3 = st.columns(3)
    summary_col_1.metric("Rows", f"{len(clean_df):,}")
    summary_col_2.metric("Detected Frequency", frequency)
    summary_col_3.metric("Latest Sales", f"{clean_df['sales'].iloc[-1]:,.2f}")

    st.subheader("Data Preview")
    st.dataframe(clean_df.tail(15), use_container_width=True)

    with st.form("forecast_form"):
        control_col_1, control_col_2 = st.columns(2)
        model_name = control_col_1.selectbox(
            "Model Selection",
            options=get_available_models(),
            index=0,
        )
        horizon = control_col_2.number_input(
            "Forecast Horizon",
            min_value=1,
            max_value=90,
            value=14,
            step=1,
            help="Number of future time periods to forecast.",
        )
        run_forecast = st.form_submit_button("Run Forecast", type="primary")

    if not run_forecast:
        st.caption(
            "The trained model artifact is saved to `models/forecasting_model.pkl` after each run."
        )
        return

    try:
        with st.spinner("Training the model and generating the forecast..."):
            artifacts = train_forecasting_model(
                clean_df=clean_df,
                model_name=model_name,
                model_path=MODEL_PATH,
            )
            forecast_df = generate_forecast(
                history_df=clean_df,
                artifacts=artifacts,
                horizon=int(horizon),
            )
    except Exception as exc:  # pragma: no cover - UI handling
        LOGGER.exception("Forecast execution failed.")
        st.error(f"Forecast generation failed: {exc}")
        st.stop()

    st.success("Forecast completed successfully.")

    metric_col_1, metric_col_2, metric_col_3 = st.columns(3)
    metric_col_1.metric("Validation MAE", f"{artifacts.metrics['mae']:.2f}")
    metric_col_2.metric("Validation RMSE", f"{artifacts.metrics['rmse']:.2f}")
    metric_col_3.metric("Forecasted Periods", f"{len(forecast_df):,}")

    st.subheader("Forecast Visualization")
    st.plotly_chart(
        create_forecast_figure(clean_df, forecast_df),
        use_container_width=True,
    )

    st.subheader("Forecast Output")
    display_df = forecast_df.copy()
    display_df[["forecast", "lower_bound", "upper_bound"]] = display_df[
        ["forecast", "lower_bound", "upper_bound"]
    ].round(2)
    st.dataframe(display_df, use_container_width=True)

    st.download_button(
        label="Download Forecast CSV",
        data=forecast_df.to_csv(index=False).encode("utf-8"),
        file_name="sales_forecast.csv",
        mime="text/csv",
    )

    st.caption(
        "Confidence bounds are estimated from validation residuals and widen gradually across the "
        "forecast horizon."
    )


if __name__ == "__main__":
    main()
