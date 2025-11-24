# SET50 Stock Price Prediction Pipeline

CLI tool to predict next-day closing prices for SET50 tickers with reproducible, leakage-aware training and evaluation.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

## Usage

Train and generate artifacts (metrics + plots):

```bash
python train.py --ticker PTT.BK --model rf --epochs 200
```

Key flags:
- `--model`: `linear`, `ridge`, `lasso`, `rf` (default `rf`)
- `--epochs`: Trees for RF or max iterations cap for linear variants.
- `--alpha`: Regularization strength for ridge/lasso.
- `--scaler`: `standard`, `minmax`, or `none` (fit on train only).
- `--train-ratio`: Time-based split ratio (default 0.8).
- `--years`: Years of daily history to download (default 5).

Outputs land in `experiments/<ticker>_<timestamp>/`:
- `metrics.json`
- `forecast_plot.png` (test actual vs predicted)
- `residual_plot.png`
- `feature_importance.png`

## Pipeline

1. **Data**: Download 5y+ daily OHLCV via `yfinance`; interpolate/forward/back-fill missing values.
2. **Features**: RSI(14), MACD (+signal/hist), Bollinger Bands (20), SMA20, EMA20, lagged Close/Volume (1,2), Return_Lag1, DayOfWeek, Month.
3. **Target**: `Target = Close.shift(-1)` (next-day close). Rows with NaNs from indicator windows are dropped.
4. **Split**: Strict time-based 80/20 (no shuffling).
5. **Scaling**: Standard or MinMax on features only; fit on train, transform test.
6. **Models**: Linear Regression, Ridge/Lasso (optional regularization), Random Forest (configurable depth/trees).
7. **Evaluation**: RMSE, MAE, MAPE, Directional Accuracy vs same-day Close baseline; naive RMSE (predict t+1 = t).
8. **Diagnostics**: Forecast, residuals, and top-10 feature importance plots.

## Notes on Stationarity

Linear regression assumes stationary relationships; raw stock prices are generally non-stationary, so coefficients can drift and degrade. Indicators, lags, and returns help, but consider differencing/log-returns or more flexible models when residuals show autocorrelation or heteroscedasticity.

## Troubleshooting

- Empty data: check ticker suffix (e.g., `.BK`) and network access.
- Flat metrics: ensure enough history (>= 200 rows after feature windows).
- Overfitting signs: increase `--max-depth` moderation, reduce `--epochs` for RF, or switch to ridge/lasso.
