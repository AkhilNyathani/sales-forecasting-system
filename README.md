# Sales Forecasting System

## Overview

This project is a production-ready sales forecasting application built with Python and Streamlit. It accepts historical sales data in CSV format, cleans and standardizes the time series, trains a forecasting model, generates future sales predictions, and visualizes the results with interactive charts.

The app is designed to run out-of-the-box with a bundled sample dataset and does not require a backend service for its core workflow.

Live Demo Link: https://sales-forecasting-system.streamlit.app/

## Features

- CSV upload for historical sales data
- Automatic preprocessing for date parsing, sorting, duplicate handling, and missing value treatment
- Forecasting with two model options:
  - Linear Regression baseline
  - XGBoost advanced model when `xgboost` is installed
- Validation metrics including MAE and RMSE
- Forecast output with estimated 95% prediction intervals
- Interactive Plotly visualization for historical and future sales
- Saved trained model artifact in `models/forecasting_model.pkl`
- Bundled sample dataset for immediate testing

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit
- XGBoost

## Project Structure

```
sales-forecasting-system/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── sample_sales.csv
│
├── models/
│   └── forecasting_model.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   ├── predict.py
│   └── utils.py
│
└── notebooks/
    └── exploration.ipynb
```

## Installation Steps

# Clone the repository

git clone https://github.com/AkhilNyathani/sales-forecasting-system.git
cd loan-approval-ml

# Create a virtual environment

python -m venv .venv

# Linux / Mac
source .venv/bin/activate

# Windows (PowerShell)
.venv\Scripts\Activate

# Install dependencies

pip install -r requirements.txt

# Running the Project

streamlit run app.py


## Usage Instructions

1. Launch the Streamlit app with `streamlit run app.py`.
2. Upload a CSV file containing `date` and `sales` columns, or use the included sample dataset.
3. Review the cleaned data preview shown in the app.
4. Choose a model from the dropdown.
5. Enter the number of future periods to forecast.
6. Click `Run Forecast`.
7. Inspect the validation metrics, forecast table, and interactive chart.
8. Download the forecast as a CSV if needed.

## Sample Output Explanation

- `Forecast`: The predicted sales value for each future period.
- `Lower Bound`: The lower end of the estimated 95% prediction interval.
- `Upper Bound`: The upper end of the estimated 95% prediction interval.
- `Validation MAE`: Mean Absolute Error on the holdout validation window.
- `Validation RMSE`: Root Mean Squared Error on the holdout validation window.

The chart overlays historical observations with forecasted values and shades the estimated uncertainty band so trends and likely ranges are easy to interpret.

## Future Improvements

- Add support for multiple target segments such as region or product category
- Add automated hyperparameter tuning for the advanced model
- Add native ARIMA or Prophet support for more classical time-series workflows
- Add API endpoints with FastAPI for production serving
- Add model monitoring and drift detection

## Screenshots

- Dashboard Placeholder: Add a screenshot of the Streamlit home screen here
- Forecast Placeholder: Add a screenshot of the forecast table and chart here

## Notes

- The app expects a CSV with `date` and `sales` columns.
- The model artifact is overwritten every time a new forecast is generated.
- If `xgboost` is unavailable in the environment, the app automatically keeps the Linear Regression option available.
