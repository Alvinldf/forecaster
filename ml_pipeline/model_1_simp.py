import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Input
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from statsmodels.tsa.stattools import grangercausalitytests
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 1. DATA FETCHING - Copper + Related Market Features
# ============================================================
def fetch_market_data(period="5y", interval="1d"):
    """
    Fetches copper futures and related market variables.
    The paper uses: exchange rates, oil, gold, silver, 
    stock indices, and other commodities.
    """
    tickers = {
        "Copper": "HG=F",
        "USD_Index": "DX-Y.NYB",   # US Dollar Index (exchange rate proxy)
        "Oil_WTI": "CL=F",         # Crude Oil WTI
        "Gold": "GC=F",            # Gold Futures
        "Silver": "SI=F",          # Silver Futures
        "SP500": "^GSPC",          # S&P 500
        "Aluminum": "ALI=F",       # Aluminum Futures
    }

    print("Fetching market data from Yahoo Finance...")
    all_data = {}
    for name, ticker in tickers.items():
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            if not data.empty:
                # Flatten multi-level columns if present
                if isinstance(data.columns, pd.MultiIndex):
                    data.columns = data.columns.get_level_values(0)
                all_data[name] = data["Close"]
                print(f"  ✓ {name} ({ticker}): {len(data)} rows")
            else:
                print(f"  ✗ {name} ({ticker}): No data")
        except Exception as e:
            print(f"  ✗ {name} ({ticker}): Error - {e}")

    # Use pd.concat instead of pd.DataFrame to handle Series objects
    df = pd.concat(all_data, axis=1)
    df.dropna(inplace=True)
    df.index.name = "Date"
    print(f"\nCombined dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df


# ============================================================
# 2. GRANGER CAUSALITY TEST - Feature Selection
# ============================================================
def granger_feature_selection(df, target="Copper", max_lag=5, significance=0.05):
    """
    Uses Granger Causality to select features that have
    predictive power for copper prices (as described in the paper).
    """
    print(f"\n--- Granger Causality Test (target={target}, max_lag={max_lag}) ---")
    selected_features = [target]

    for col in df.columns:
        if col == target:
            continue
        try:
            test_data = df[[target, col]].dropna()
            result = grangercausalitytests(test_data, maxlag=max_lag, verbose=False)
            # Get the minimum p-value across all lags
            min_p = min([result[lag][0]["ssr_ftest"][1] for lag in range(1, max_lag + 1)])
            if min_p < significance:
                selected_features.append(col)
                print(f"  ✓ {col}: p-value={min_p:.6f} (SELECTED)")
            else:
                print(f"  ✗ {col}: p-value={min_p:.6f} (REJECTED)")
        except Exception as e:
            print(f"  ✗ {col}: Error - {e}")

    print(f"\nSelected features: {selected_features}")
    return selected_features


# ============================================================
# 3. DATA PREPROCESSING
# ============================================================
def preprocess_data(df, features, lookback=20, train_ratio=0.8):
    """
    Scales data and creates sequences for LSTM/CNN input.
    The paper uses a lookback window and MinMax scaling.
    """
    data = df[features].values
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback : i, :])  # All features
        y.append(scaled_data[i, 0])                  # Target: Copper (col 0)

    X, y = np.array(X), np.array(y)

    # Train/Test split (temporal, no shuffle)
    split = int(len(X) * train_ratio)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    print(f"\nData shapes:")
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    return X_train, X_test, y_train, y_test, scaler


# ============================================================
# 4. MODEL ARCHITECTURES (from the paper)
# ============================================================

# --- 4a. LSTM Model ---
def build_lstm(lookback, n_features, dropout_rate=0.2):
    """LSTM model as described in the paper."""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(lookback, n_features)),
        Dropout(dropout_rate),
        LSTM(32, return_sequences=False),
        Dropout(dropout_rate),
        Dense(16, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


# --- 4b. CNN Model ---
def build_cnn(lookback, n_features, dropout_rate=0.2):
    """CNN model for time series as described in the paper."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu",
               input_shape=(lookback, n_features)),
        MaxPooling1D(pool_size=2),
        Conv1D(filters=32, kernel_size=3, activation="relu"),
        Dropout(dropout_rate),
        Flatten(),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


# --- 4c. CNN-LSTM Hybrid Model ---
def build_cnn_lstm(lookback, n_features, dropout_rate=0.2):
    """CNN-LSTM hybrid model as described in the paper."""
    model = Sequential([
        Conv1D(filters=64, kernel_size=3, activation="relu",
               input_shape=(lookback, n_features)),
        MaxPooling1D(pool_size=2),
        LSTM(50, return_sequences=False),
        Dropout(dropout_rate),
        Dense(32, activation="relu"),
        Dense(1),
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mse")
    return model


# ============================================================
# 5. MONTE CARLO DROPOUT - Uncertainty Estimation
# ============================================================
def mc_dropout_predict(model, X, n_simulations=100):
    """
    Monte Carlo Dropout for uncertainty estimation.
    The paper uses dropout at inference time to generate
    a distribution of predictions.
    """
    predictions = []
    for _ in range(n_simulations):
        # training=True keeps dropout active during inference
        pred = model(X, training=True).numpy().flatten()
        predictions.append(pred)

    predictions = np.array(predictions)
    mean_pred = predictions.mean(axis=0)
    std_pred = predictions.std(axis=0)

    return mean_pred, std_pred


# ============================================================
# 6. EVALUATION METRICS
# ============================================================
def evaluate_model(y_true, y_pred, model_name="Model"):
    """Calculates RMSE and MAE as used in the paper."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    print(f"\n{model_name} Results:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE:  {mae:.6f}")
    return rmse, mae


# ============================================================
# 7. INVERSE TRANSFORM PREDICTIONS
# ============================================================
def inverse_transform_target(scaler, y_scaled, n_features):
    """Inverse transforms scaled target values back to original scale."""
    dummy = np.zeros((len(y_scaled), n_features))
    dummy[:, 0] = y_scaled
    return scaler.inverse_transform(dummy)[:, 0]


# ============================================================
# 8. PLOTTING
# ============================================================
def plot_results(y_true, predictions_dict, title="Copper Price Forecast"):
    """Plots actual vs predicted for all models."""
    plt.figure(figsize=(14, 6))
    plt.plot(y_true, label="Actual", color="black", linewidth=2)

    colors = ["blue", "red", "green"]
    for i, (name, preds) in enumerate(predictions_dict.items()):
        plt.plot(preds, label=name, color=colors[i % len(colors)], alpha=0.7)

    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Copper Price (USD/lb)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("copper_forecast_results.png", dpi=150)
    plt.show()
    print("Plot saved to copper_forecast_results.png")


def plot_uncertainty(y_true, mean_pred, std_pred, title="MC Dropout Uncertainty"):
    """Plots predictions with confidence intervals."""
    plt.figure(figsize=(14, 6))
    x = range(len(y_true))
    plt.plot(x, y_true, label="Actual", color="black", linewidth=2)
    plt.plot(x, mean_pred, label="MC Dropout Mean", color="blue", alpha=0.8)
    plt.fill_between(
        x,
        mean_pred - 2 * std_pred,
        mean_pred + 2 * std_pred,
        alpha=0.2,
        color="blue",
        label="95% Confidence Interval",
    )
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Copper Price (USD/lb)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("copper_uncertainty.png", dpi=150)
    plt.show()
    print("Plot saved to copper_uncertainty.png")


# ============================================================
# MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    # --- Step 1: Fetch Data ---
    df = fetch_market_data(period="5y")

    # --- Step 2: Feature Selection via Granger Causality ---
    selected = granger_feature_selection(df, target="Copper", max_lag=5)

    # --- Step 3: Preprocess ---
    lookback = 20
    X_train, X_test, y_train, y_test, scaler = preprocess_data(
        df, selected, lookback=lookback, train_ratio=0.8
    )
    n_features = X_train.shape[2]

    # --- Step 4: Train Models ---
    early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    epochs = 50
    batch_size = 32

    models = {
        "LSTM": build_lstm(lookback, n_features),
        "CNN": build_cnn(lookback, n_features),
        "CNN-LSTM": build_cnn_lstm(lookback, n_features),
    }

    results = {}
    predictions_dict = {}

    for name, model in models.items():
        print(f"\n{'='*50}")
        print(f"Training {name}...")
        print(f"{'='*50}")

        model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.15,
            callbacks=[early_stop],
            verbose=1,
        )

        # Predict
        y_pred_scaled = model.predict(X_test).flatten()

        # Inverse transform
        y_pred = inverse_transform_target(scaler, y_pred_scaled, n_features)
        y_actual = inverse_transform_target(scaler, y_test, n_features)

        # Evaluate
        rmse, mae = evaluate_model(y_actual, y_pred, model_name=name)
        results[name] = {"RMSE": rmse, "MAE": mae}
        predictions_dict[name] = y_pred

    # --- Step 5: Monte Carlo Dropout Uncertainty (on best model) ---
    print(f"\n{'='*50}")
    print("Monte Carlo Dropout Uncertainty Estimation (LSTM)")
    print(f"{'='*50}")

    mc_mean_scaled, mc_std_scaled = mc_dropout_predict(
        models["LSTM"], X_test, n_simulations=100
    )
    mc_mean = inverse_transform_target(scaler, mc_mean_scaled, n_features)
    mc_std = inverse_transform_target(scaler, mc_std_scaled, n_features)

    # --- Step 6: Summary ---
    print(f"\n{'='*50}")
    print("SUMMARY OF RESULTS")
    print(f"{'='*50}")
    summary_df = pd.DataFrame(results).T
    print(summary_df.to_string())

    # --- Step 7: Plot ---
    y_actual = inverse_transform_target(scaler, y_test, n_features)
    plot_results(y_actual, predictions_dict)
    plot_uncertainty(y_actual, mc_mean, mc_std)

    print("\nDone!")