import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.statsmodels
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from influxdb_client import InfluxDBClient
import os

# 1. Point to the config variables
from config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET

# 2. Point to the Docker container
tracking_uri = "http://localhost:5000"
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

def train_arima(df):
    """Trains a simple ARIMA model and logs to MLflow."""
    print("Training ARIMA Model...")
    with mlflow.start_run(run_name="ARIMA_Baseline"):
        # Simple ARIMA configuration (1, 1, 1)
        order = (1, 1, 1)
        model = ARIMA(df['close'], order=order)
        fitted_model = model.fit()
        
        # Calculate a simple metric (In-sample error)
        predictions = fitted_model.predict(start=1, end=len(df)-1)
        rmse = mean_squared_error(df['close'].iloc[1:], predictions) ** 0.5
        # Log to MLflow
        mlflow.log_param("model_type", "ARIMA")
        mlflow.log_param("order", str(order))
        mlflow.log_metric("rmse", rmse)
        mlflow.statsmodels.log_model(fitted_model, "model")
        
        print(f"ARIMA Logged! RMSE: {rmse:.4f}")

def train_random_forest(df):
    """Trains a simple Random Forest and logs to MLflow."""
    print("Training Random Forest Model...")
    with mlflow.start_run(run_name="RF_Baseline"):
        # Create a simple feature (yesterday's price) and target (today's price)
        rf_df = df.copy()
        rf_df['lag_1'] = rf_df['close'].shift(1)
        rf_df.dropna(inplace=True)
        
        X = rf_df[['lag_1']]
        y = rf_df['close']
        
        # Train model
        estimators = 50
        model = RandomForestRegressor(n_estimators=estimators, random_state=42)
        model.fit(X, y)
        
        # Calculate metric
        predictions = model.predict(X)
        rmse = mean_squared_error(y, predictions) ** 0.5
        
        # Log to MLflow
        mlflow.log_param("model_type", "Random Forest")
        mlflow.log_param("n_estimators", estimators)
        mlflow.log_metric("rmse", rmse)
        mlflow.sklearn.log_model(model, "model")
        
        print(f"Random Forest Logged! RMSE: {rmse:.4f}")

if __name__ == "__main__":
    # 1. Get the data
    copper_data = fetch_training_data(ticker="HG=F")
    
    if copper_data.empty:
        print("No data found. Did you run data_ingestion.py?")
    else:
        # 2. Train and track multiple models in one script
        train_arima(copper_data)
        train_random_forest(copper_data)
        print("All models successfully trained and logged to MLflow!")