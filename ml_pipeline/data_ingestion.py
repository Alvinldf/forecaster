import pandas as pd
import yfinance as yf
import requests
from datetime import datetime
from influxdb_client import InfluxDBClient, BucketRetentionRules
from influxdb_client.client.write_api import SYNCHRONOUS
from config import (
    INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, 
    INFLUX_BUCKET, INFLUX_BUCKET_FORECAST, 
    TICKERS, BCRP_SERIES_USDPEN
)

def init_influxdb_buckets():
    """Checks if required buckets exist and creates them if they do not."""
    print("Checking InfluxDB buckets...")
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    buckets_api = client.buckets_api()
    
    existing_buckets = [b.name for b in buckets_api.find_buckets().buckets]
    
    for bucket_name in [INFLUX_BUCKET, INFLUX_BUCKET_FORECAST]:
        if bucket_name not in existing_buckets:
            print(f"Bucket '{bucket_name}' not found. Creating it now...")
            # Set retention to 0 (infinite) so historical data isn't deleted
            retention_rule = BucketRetentionRules(type="expire", every_seconds=0)
            buckets_api.create_bucket(bucket_name=bucket_name, org=INFLUX_ORG, retention_rules=[retention_rule])
            print(f"Success! '{bucket_name}' created.")
        else:
            print(f"Bucket '{bucket_name}' already exists.")
            
    client.close()

import time

def fetch_bcrp_historical():
    """Fetches ALL historical USD/PEN data from BCRP in 4-year chunks."""
    print(f"Fetching ALL historical USD/PEN data from BCRP (Series: {BCRP_SERIES_USDPEN})...")
    
    # BCRP daily data for this series starts in 1997.
    # We step through the years in blocks of 4.
    chunk_starts = list(range(1997, 2027, 4))
    
    for i in range(len(chunk_starts)):
        start_year = chunk_starts[i]
        # Calculate the end of the chunk, capping at 2026
        end_year = chunk_starts[i+1] - 1 if i + 1 < len(chunk_starts) else 2026
        
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"
        
        url = f"https://estadisticas.bcrp.gob.pe/estadisticas/series/api/{BCRP_SERIES_USDPEN}/json/{start_date}/{end_date}"
        print(f" -> Fetching chunk: {start_date} to {end_date}...")
        
        headers = {
            "Accept": "application/json",
            "User-Agent": "Mozilla/5.0"
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            periods = data.get("periods", [])
            if not periods:
                print(f"    No data returned for chunk {start_year}-{end_year}.")
                continue
                
            records = []
            for p in periods:
                val = float(p["values"][0])
                try:
                    dt = pd.to_datetime(p["name"], format="%d.%b.%y", errors='coerce')
                    if pd.isna(dt):
                        dt = pd.to_datetime(p["name"], dayfirst=True)
                except:
                    continue
                    
                records.append({
                    "time": dt.tz_localize('UTC') if dt.tz is None else dt.tz_convert('UTC'),
                    "close": val,
                    "ticker": "USDPEN=X",
                    "provider": "BCRP"
                })
                
            df_bcrp = pd.DataFrame(records).set_index("time")
            write_to_influx(df_bcrp, "market_price")
            print(f"    Success! {len(df_bcrp)} records ingested.")
            
            # Be polite to the BCRP servers and wait 2 seconds before the next chunk
            time.sleep(2)
            
        except Exception as e:
            print(f"    Error fetching chunk {start_year}-{end_year}: {e}")
def fetch_yfinance_historical(ticker_symbol):
    """Fetches 10 years of daily historical data from Yahoo Finance."""
    print(f"Fetching historical data for {ticker_symbol} from Yahoo Finance...")
    
    ticker = yf.Ticker(ticker_symbol)
    # 10 years of daily data gives ML models plenty of training context
    df = ticker.history(period="max", interval="1d")
    
    if df.empty:
        print(f"No data found for {ticker_symbol}.")
        return
        
    df.dropna(inplace=True)
    
    df['ticker'] = ticker_symbol
    df['provider'] = 'yfinance'
    
    df.rename(columns={
        "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"
    }, inplace=True)
    
    columns_to_keep = ["open", "high", "low", "close", "volume", "ticker", "provider"]
    df = df[[col for col in columns_to_keep if col in df.columns]]
    
    if df.index.tz is not None:
        df.index = df.index.tz_convert('UTC')
    else:
        df.index = df.index.tz_localize('UTC')

    write_to_influx(df, "market_price")
    print(f"Success! {len(df)} records ingested for {ticker_symbol}.")

def write_to_influx(df, measurement_name):
    """Helper function to write DataFrames to InfluxDB."""
    # Added timeout=30000 to prevent large datasets from crashing the connection
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG, timeout=30000)
    write_api = client.write_api(write_options=SYNCHRONOUS)
    
    write_api.write(
        bucket=INFLUX_BUCKET, 
        org=INFLUX_ORG, 
        record=df,
        data_frame_measurement_name=measurement_name,
        data_frame_tag_columns=["ticker", "provider"]
    )
    client.close()

if __name__ == "__main__":
    print("--- Starting Historical Data Ingestion Pipeline ---")
    
    # 1. Ensure databases exist
    init_influxdb_buckets()
    
    # 2. Ingest 10 years of Macro FX data
    fetch_bcrp_historical()
    
    # 3. Ingest 10 years of Commodity data
    for ticker in TICKERS:
        fetch_yfinance_historical(ticker)
        
    print("--- Pipeline Execution Complete ---")