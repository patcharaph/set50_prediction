import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler


@dataclass
class Dataset:
    features: pd.DataFrame
    target: pd.Series


def fetch_price_history(ticker: str, years: int = 5) -> pd.DataFrame:
    """Download daily OHLCV data for the given ticker."""
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=365 * years + 5)
    df = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df.empty:
        raise ValueError(f"No data returned for ticker {ticker}")
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df = df.sort_index()
    return df


def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Interpolate missing values and drop any leading/trailing gaps."""
    filled = df.interpolate(method="time").ffill().bfill()
    return filled


def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators, lags, date features, and target."""
    data = df.copy()
    data["Return"] = data["Close"].pct_change()

    data["RSI14"] = compute_rsi(data["Close"], 14)

    ema12 = data["Close"].ewm(span=12, adjust=False).mean()
    ema26 = data["Close"].ewm(span=26, adjust=False).mean()
    data["MACD"] = ema12 - ema26
    data["MACD_Signal"] = data["MACD"].ewm(span=9, adjust=False).mean()
    data["MACD_Hist"] = data["MACD"] - data["MACD_Signal"]

    rolling20 = data["Close"].rolling(window=20)
    data["BB_Mid"] = rolling20.mean()
    bb_std = rolling20.std()
    data["BB_Upper"] = data["BB_Mid"] + 2 * bb_std
    data["BB_Lower"] = data["BB_Mid"] - 2 * bb_std
    data["SMA20"] = data["Close"].rolling(window=20).mean()
    data["EMA20"] = data["Close"].ewm(span=20, adjust=False).mean()

    for lag in (1, 2):
        data[f"Close_Lag{lag}"] = data["Close"].shift(lag)
        data[f"Volume_Lag{lag}"] = data["Volume"].shift(lag)
    data["Return_Lag1"] = data["Return"].shift(1)

    data["DayOfWeek"] = data.index.dayofweek
    data["Month"] = data.index.month

    data["Target"] = data["Close"].shift(-1)
    engineered = data.dropna()
    return engineered


def build_datasets(df: pd.DataFrame, train_ratio: float = 0.8) -> Tuple[Dataset, Dataset]:
    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    if train_df.empty or test_df.empty:
        raise ValueError("Train/test split produced an empty set; adjust train_ratio or gather more data.")

    feature_cols = [col for col in df.columns if col not in {"Target"}]
    X_train = train_df[feature_cols]
    y_train = train_df["Target"]
    X_test = test_df[feature_cols]
    y_test = test_df["Target"]
    return Dataset(X_train, y_train), Dataset(X_test, y_test)


def scale_features(
    train: Dataset, test: Dataset, scaler_type: str = "standard"
) -> Tuple[Dataset, Dataset, object]:
    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler:
        scaler.fit(train.features)
        X_train_scaled = pd.DataFrame(
            scaler.transform(train.features),
            index=train.features.index,
            columns=train.features.columns,
        )
        X_test_scaled = pd.DataFrame(
            scaler.transform(test.features),
            index=test.features.index,
            columns=test.features.columns,
        )
    else:
        X_train_scaled, X_test_scaled = train.features, test.features

    return (
        Dataset(X_train_scaled, train.target),
        Dataset(X_test_scaled, test.target),
        scaler,
    )


def choose_model(
    model_name: str, epochs: int, alpha: float, max_depth: int, random_state: int
):
    if model_name == "linear":
        return LinearRegression()
    if model_name == "ridge":
        return Ridge(alpha=alpha, random_state=random_state, max_iter=max(epochs, 1000))
    if model_name == "lasso":
        return Lasso(alpha=alpha, random_state=random_state, max_iter=max(epochs, 1000))
    if model_name == "rf":
        return RandomForestRegressor(
            n_estimators=epochs,
            max_depth=max_depth if max_depth > 0 else None,
            random_state=random_state,
            n_jobs=-1,
        )
    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_predictions(
    y_true: pd.Series, y_pred: np.ndarray, reference_close: pd.Series
) -> Dict[str, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    true_direction = np.sign(y_true.values - reference_close.values)
    pred_direction = np.sign(y_pred - reference_close.values)
    directional_accuracy = float((true_direction == pred_direction).mean())

    return {
        "rmse": float(rmse),
        "mae": float(mae),
        "mape": float(mape),
        "directional_accuracy": directional_accuracy,
    }


def compute_baseline_rmse(test_close: pd.Series, target: pd.Series) -> float:
    naive_pred = test_close.values
    return float(np.sqrt(mean_squared_error(target, naive_pred)))


def plot_forecast(
    dates: pd.Index, y_true: pd.Series, y_pred: np.ndarray, output_path: Path
):
    plt.figure(figsize=(10, 4))
    plt.plot(dates, y_true, label="Actual", lw=2)
    plt.plot(dates, y_pred, label="Predicted", lw=2)
    plt.legend()
    plt.title("Next-Day Close Forecast (Test)")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_residuals(dates: pd.Index, residuals: np.ndarray, output_path: Path):
    plt.figure(figsize=(10, 4))
    plt.axhline(0, color="gray", lw=1)
    plt.plot(dates, residuals, marker="o", linestyle="-", ms=3)
    plt.title("Residuals (Actual - Predicted)")
    plt.xlabel("Date")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_feature_importance(
    model, feature_names: List[str], output_path: Path, top_k: int = 10
):
    if hasattr(model, "feature_importances_"):
        importance = pd.Series(model.feature_importances_, index=feature_names)
    elif hasattr(model, "coef_"):
        coeffs = model.coef_
        if coeffs.ndim > 1:
            coeffs = coeffs[0]
        importance = pd.Series(np.abs(coeffs), index=feature_names)
    else:
        return

    top_imp = importance.sort_values(ascending=False).head(top_k)
    plt.figure(figsize=(8, 5))
    top_imp.iloc[::-1].plot(kind="barh")
    plt.title("Top Feature Importance")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def persist_metrics(metrics: Dict[str, float], output_path: Path):
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SET50 Next-Day Close prediction pipeline"
    )
    parser.add_argument("--ticker", required=True, help="Ticker symbol, e.g., PTT.BK")
    parser.add_argument(
        "--years", type=int, default=5, help="Years of daily history to download"
    )
    parser.add_argument(
        "--model",
        choices=["linear", "ridge", "lasso", "rf"],
        default="rf",
        help="Model type",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs / trees (for RF) or iterations ceiling",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularization strength for ridge/lasso",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=8,
        help="Max depth for random forest (0 means unlimited)",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Proportion of data used for training",
    )
    parser.add_argument(
        "--scaler",
        choices=["standard", "minmax", "none"],
        default="standard",
        help="Feature scaling method",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments",
        help="Directory to store plots and metrics",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)

    print(f"Fetching {args.years} years of data for {args.ticker}...")
    raw = fetch_price_history(args.ticker, years=args.years)
    cleaned = handle_missing(raw)
    engineered = engineer_features(cleaned)
    if engineered.empty:
        raise ValueError("Not enough data after feature engineering to proceed.")

    train_ds, test_ds = build_datasets(engineered, train_ratio=args.train_ratio)
    train_scaled, test_scaled, scaler = scale_features(
        train_ds, test_ds, scaler_type=args.scaler
    )

    model = choose_model(
        args.model, epochs=args.epochs, alpha=args.alpha, max_depth=args.max_depth, random_state=args.seed
    )
    print(f"Training {args.model} model...")
    model.fit(train_scaled.features, train_scaled.target)

    y_pred = model.predict(test_scaled.features)
    baseline_rmse = compute_baseline_rmse(
        test_ds.features["Close"], test_ds.target
    )
    metrics = evaluate_predictions(
        test_scaled.target, y_pred, reference_close=test_ds.features["Close"]
    )
    metrics["baseline_rmse"] = baseline_rmse

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"{args.ticker}_{timestamp}"
    os.makedirs(run_dir, exist_ok=True)

    forecast_path = run_dir / "forecast_plot.png"
    residual_path = run_dir / "residual_plot.png"
    importance_path = run_dir / "feature_importance.png"
    metrics_path = run_dir / "metrics.json"

    plot_forecast(test_scaled.features.index, test_scaled.target, y_pred, forecast_path)
    residuals = test_scaled.target.values - y_pred
    plot_residuals(test_scaled.features.index, residuals, residual_path)
    plot_feature_importance(model, list(train_scaled.features.columns), importance_path)
    persist_metrics(metrics, metrics_path)

    print("Metrics:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")
    print(f"Baseline RMSE: {baseline_rmse:.4f}")
    print(f"Artifacts saved to {run_dir}")


if __name__ == "__main__":
    main()
