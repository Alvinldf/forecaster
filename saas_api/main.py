import os
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session
from dotenv import load_dotenv
from influxdb_client import InfluxDBClient

import database

# --- 1. Load Environment Variables ---
# Go up one directory to find the root .env file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')
load_dotenv(dotenv_path=ENV_PATH)

INFLUX_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_ORG = os.getenv("INFLUXDB_ORG")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET")

# --- 2. Initialize FastAPI ---
app = FastAPI(
    title="Macroeconomic Forecaster API",
    description="Backend engine for commodity and FX predictive alerts.",
    version="1.0.0"
)
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3001"], # Your Next.js port
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 3. SQLite Dependency ---
def get_db():
    db = database.SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- 4. API Endpoints ---

@app.get("/")
def read_root():
    return {"status": "online", "message": "Forecaster API is running."}

@app.get("/health")
def health_check():
    return {"api": "healthy", "database": "connected"}

@app.get("/clients/")
def get_clients(db: Session = Depends(get_db)):
    """Fetches all human user profiles from the SQLite database."""
    clients = db.query(database.Client).all()
    return {"clients": clients}

@app.get("/api/latest-price/{ticker}")
def get_latest_price(ticker: str):
    """
    Reaches into InfluxDB and pulls the most recent closing price
    for a given ticker (e.g., 'HG=F' for Copper or 'USDPEN=X' for FX).
    """
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        # Flux query: get the last 30 days, filter for the ticker, grab the last one
        query = f'''
            from(bucket: "{INFLUX_BUCKET}")
            |> range(start: -30d)
            |> filter(fn: (r) => r._measurement == "market_price")
            |> filter(fn: (r) => r.ticker == "{ticker}")
            |> filter(fn: (r) => r._field == "close")
            |> sort(columns: ["_time"], desc: true)
            |> limit(n: 1)
        '''
        
        result = query_api.query(org=INFLUX_ORG, query=query)
        client.close()
        
        # Parse the result
        if result and len(result) > 0 and len(result[0].records) > 0:
            record = result[0].records[0]
            return {
                "ticker": ticker,
                "latest_close": record.get_value(),
                "timestamp": record.get_time().isoformat()
            }
        
        raise HTTPException(status_code=404, detail=f"No data found for {ticker} in the last 30 days.")
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


import pandas as pd
import mlflow.statsmodels
from influxdb_client import InfluxDBClient

# --- New: MLflow Configuration ---
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

@app.get("/api/forecast/{ticker}")
def get_price_forecast(ticker: str):
    """
    Downloads the latest model from MLflow, gets recent data from InfluxDB,
    and generates a 30-day forecast.
    """
    try:
        # 1. Load the "Latest" version of your ARIMA model from MLflow
        # We use 'models:/' followed by the name you used in train.py
        model_name = "ARIMA_Baseline" 
        model_uri = f"models:/Copper_SaaS_Final/latest" 
        
        # For this tracer bullet, we can also point directly to the run ID
        # if you haven't registered the model yet. Let's use the simplest method:
        model = mlflow.statsmodels.load_model(model_uri)
        
        # 2. Get the latest data from InfluxDB to provide context
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        # We need at least enough data to satisfy the ARIMA lags (e.g., last 30 days)
        query = f'''
            from(bucket: "{INFLUX_BUCKET}")
            |> range(start: -30d)
            |> filter(fn: (r) => r._measurement == "market_price")
            |> filter(fn: (r) => r.ticker == "{ticker}")
            |> filter(fn: (r) => r._field == "close")
        '''
        result = query_api.query(org=INFLUX_ORG, query=query)
        client.close()

        # 3. Generate the 30-day forecast
        # ARIMA models in statsmodels have a .forecast() method
        forecast_steps = 30
        prediction = model.forecast(steps=forecast_steps)
        
        return {
            "ticker": ticker,
            "forecast_days": forecast_steps,
            "predictions": prediction.tolist(),
            "model_source": model_uri
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Error: {str(e)}")