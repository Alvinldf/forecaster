import os
from dotenv import load_dotenv

# Get the absolute path to the .env file in the parent 'forecaster' directory
# This ensures it works no matter where you run the script from
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(BASE_DIR, '.env')

load_dotenv(dotenv_path=ENV_PATH)

# Export the variables so other scripts can import them
INFLUX_URL = os.getenv("INFLUXDB_URL", "http://localhost:8086")
INFLUX_TOKEN = os.getenv("INFLUXDB_TOKEN")
INFLUX_ORG = os.getenv("INFLUXDB_ORG")
INFLUX_BUCKET = os.getenv("INFLUXDB_BUCKET")
INFLUX_BUCKET_FORECAST = os.getenv("INFLUXDB_BUCKET_FORECAST")

# Tickers for Yahoo Finance
TICKERS = ["BTC-USD", "GC=F", "HG=F", "SI=F"]

# BCRP Series for USD/PEN
BCRP_SERIES_USDPEN = "PD04638PD"