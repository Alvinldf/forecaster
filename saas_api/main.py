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
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET", "MarketData")
INFLUX_BUCKET_FORECAST = os.getenv("INFLUXDB_BUCKET_FORECAST", "ForecastData")

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
    Fetches the latest pre-computed CNN-LSTM forecast signal from InfluxDB.
    This architecture prevents the web server from hanging during heavy ML inference.
    """
    try:
        client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        query_api = client.query_api()
        
        # Query the dedicated Forecast bucket para T+1
        # Filtramos 'ANCHOR' y buscamos desde el instante actual hacia adelante
        # Re-agrupamos todas las sub-tablas (Influx crea tablas por cada tag distinto de Señal/Urgencia) 
        # antes de ordenar para que el T+1 global gane.
        query = f'''
            from(bucket: "{INFLUX_BUCKET_FORECAST}")
            |> range(start: now(), stop: 14d)
            |> filter(fn: (r) => r._measurement == "forecast_signal")
            |> filter(fn: (r) => r.ticker == "{ticker}")
            |> filter(fn: (r) => r.is_backtest == "False")
            |> filter(fn: (r) => r.signal != "ANCHOR")
            |> pivot(rowKey:["_time"], columnKey: ["_field"], valueColumn: "_value")
            |> group()
            |> sort(columns: ["_time"], desc: false)
            |> limit(n: 1)
        '''
        
        result = query_api.query(org=INFLUX_ORG, query=query)
        client.close()

        if not result or len(result) == 0 or len(result[0].records) == 0:
            raise HTTPException(status_code=404, detail=f"No recent forecasts found for {ticker}")

        record = result[0].records[0]
        
        return {
            "ticker": ticker,
            "forecast_date": record.get_time().isoformat(),
            "signal": record.values.get("signal", "UNKNOWN"),
            "urgency": record.values.get("urgency", "UNKNOWN"),
            "expected_price": record.values.get("expected_price"),
            "expected_return_pct": record.values.get("expected_return_pct"),
            "upward_probability": record.values.get("upward_probability"),
            "probability_uncertainty": record.values.get("probability_uncertainty"),
            "model_type": record.values.get("model_type", "CNN-LSTM_MultiTask")
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference Data Fetch Error: {str(e)}")