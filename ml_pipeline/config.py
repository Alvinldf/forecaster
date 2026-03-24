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
TICKERS = [
    "BTC-USD",
    # --- TARGET ---
    "SI=F",           # Silver Futures (Plata)

    # --- METALES (Grupo 2: Relación de Sustitución/Oferta) ---
    "GC=F",           # Gold Futures (Oro)
    "HG=F",           # Copper Futures (Cobre - Insumo Industrial)
    "PL=F",           # Platinum Futures (Platino)
    "PA=F",           # Palladium Futures (Paladio - Automotriz)
    "ALI=F",          # Aluminum Futures (Aluminio)

    # --- MACRO / MONETARIO (Grupo 3: Costo de Oportunidad) ---
    "DX-Y.NYB",       # US Dollar Index (DXY)
    "SHY",            # iShares 1-3 Year Treasury Bond ETF (Tasas cortas)
    "TLT",            # iShares 20+ Year Treasury Bond ETF (Tasas largas)
    "TIP",            # iShares TIPS Bond ETF (Inflación esperada)

    # --- ENERGÍA (Grupo 1/2: Costos de Producción e Industria) ---
    "CL=F",           # Crude Oil (Petróleo)
    "NG=F",           # Natural Gas (Gas Natural)

    # --- MERCADO DE CAPITALES (Grupo 4: Sentimiento) ---
    "^GSPC",          # S&P 500 (EE.UU.)
    "^IXIC",          # NASDAQ Composite (Tecnología/Demanda Plata)
    "SLV",            # iShares Silver Trust (Flujos de inversión)
    "SIL",            # Global X Silver Miners ETF (Salud del sector minero)
    "EEM",            # MSCI Emerging Markets ETF (Demanda de países emergentes)

    # --- DIVISAS RELEVANTES ---
    "EURUSD=X",       # Euro / Dólar
    "CNY=X",          # Dólar / Yuan (China es el mayor consumidor industrial)
    "JPY=X"           # Dólar / Yen (Refugio)
]

# BCRP Series for USD/PEN
BCRP_SERIES_USDPEN = "PD04638PD"