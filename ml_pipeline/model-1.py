import pandas as pd
import numpy as np
import mlflow
import mlflow.keras
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from influxdb_client import InfluxDBClient
import os

# 1. Point to the config variables
from config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET

# 2. Point to the Docker container
tracking_uri = "http://localhost:8888"
mlflow.set_tracking_uri(tracking_uri)
os.environ['MLFLOW_TRACKING_URI'] = tracking_uri

# 3. Set Experiment
experiment_name = "Copper_SaaS_Final"
mlflow.set_experiment(experiment_name)

def fetch_training_data(ticker="HG=F", days=730):
    """Fetches the last 2 years of daily data from InfluxDB for training."""
    print(f"Fetching {days} days of training data for {ticker}...")
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()

    # Flux query to grab the closing prices
    query = f'''
        from(bucket: "{INFLUX_BUCKET}")
        |> range(start: -{days}d)
        |> filter(fn: (r) => r._measurement == "market_price")
        |> filter(fn: (r) => r.ticker == "{ticker}")
        |> filter(fn: (r) => r._field == "close")
        |> keep(columns: ["_time", "_value"])
    '''
    
    result = query_api.query(org=INFLUX_ORG, query=query)
    client.close()

    records = []
    for table in result:
        for record in table.records:
            records.append({"time": record.get_time(), "close": record.get_value()})
            
    df = pd.DataFrame(records).set_index("time")
    df.index = df.index.tz_localize(None) # Remove timezone for ML models
    df.sort_index(inplace=True)
    return df

def create_lstm_data(df, lookback=10):
    """Prepares the data for LSTM training."""
    data = df['close'].values
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i-lookback:i])
        y.append(data[i])
    X, y = np.array(X), np.array(y)
    return X, y

def train_lstm(df):
    """Trains an LSTM model and logs to MLflow."""
    print("Training LSTM Model...")
    with mlflow.start_run(run_name="LSTM_Baseline"):
        # Prepare the data
        lookback = 10
        X, y = create_lstm_data(df, lookback)
        X = X.reshape((X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
        
        # Build the LSTM model
        model = Sequential([
            LSTM(50, activation='relu', input_shape=(lookback, 1)),
            Dense(1)
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        
        # Train the model
        history = model.fit(X, y, epochs=20, batch_size=32, verbose=1)
        
        # Calculate RMSE on training data
        predictions = model.predict(X)
        rmse = mean_squared_error(y, predictions) ** 0.5
        
        # Log to MLflow
        mlflow.log_param("model_type", "LSTM")
        mlflow.log_param("lookback", lookback)
        mlflow.log_metric("rmse", rmse)
        mlflow.keras.log_model(model, "model")
        
        print(f"LSTM Logged! RMSE: {rmse:.4f}")

if __name__ == "__main__":
    # 1. Get the data
    copper_data = fetch_training_data(ticker="HG=F")
    
    if copper_data.empty:
        print("No data found. Did you run data_ingestion.py?")
    else:
        # 2. Train and track the LSTM model
        train_lstm(copper_data)
        print("LSTM model successfully trained and logged to MLflow!")