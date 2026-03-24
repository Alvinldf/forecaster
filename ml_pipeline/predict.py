import os
import warnings
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
import tensorflow as tf
from tensorflow.keras.models import load_model
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET, INFLUX_BUCKET_FORECAST
except ImportError:
    INFLUX_URL = "http://localhost:8086"
    INFLUX_TOKEN = "your-token"
    INFLUX_ORG = "your-org"
    INFLUX_BUCKET = "market_data"
    INFLUX_BUCKET_FORECAST = "market_forecast"

TARGET_TICKER = 'SI=F'
RELEVANT_TICKERS = ['SLV', 'NG=F', 'SIL', 'CL=F']
ALL_TICKERS = [TARGET_TICKER] + RELEVANT_TICKERS
WINDOW_SIZE = 10

def fetch_recent_data(tickers, days=60):
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    query_api = client.query_api()
    all_dfs = []
    
    for t in tickers:
        query = f'''
        from(bucket:"{INFLUX_BUCKET}") 
        |> range(start: -{days}d) 
        |> filter(fn:(r)=>r.ticker=="{t}" and r._field=="close") 
        |> pivot(rowKey:["_time"], columnKey:["_field"], valueColumn:"_value")
        '''
        df = query_api.query_data_frame(query)
        if not df.empty:
            df = df[['_time', 'close']].rename(columns={'_time': 'Date', 'close': t})
            df.set_index('Date', inplace=True)
            all_dfs.append(df)
            
    client.close()
    if not all_dfs:
        raise ValueError("No data returned from InfluxDB.")
    
    merged_df = pd.concat(all_dfs, axis=1).sort_index()
    return merged_df.resample('D').last().ffill().dropna()

def mc_dropout_predict_mt(model, X, n_iter=100):
    preds_cls, preds_reg = [], []
    for _ in range(n_iter):
        p_dict = model(X, training=True)
        preds_cls.append(p_dict['clasificador'].numpy())
        preds_reg.append(p_dict['magnitud'].numpy())
    return np.array(preds_cls).squeeze(), np.array(preds_reg).squeeze()

def run_prediction():
    print("🚀 Iniciando Motor de Inferencia Auto-Regresivo (MCD) a 5 días...")
    
    scaler_path = "models/scaler.joblib"
    model_path = "models/cnn_lstm_multitask.keras"
    if not os.path.exists(scaler_path) or not os.path.exists(model_path):
        print(f"Error: No se encontraron los modelos entrenados en {model_path}.")
        return

    scaler = joblib.load(scaler_path)
    model = load_model(model_path)
    print("✅ Modelos y Scaler cargados en memoria.")

    df_raw = fetch_recent_data(ALL_TICKERS, days=60)
    if len(df_raw) <= WINDOW_SIZE + 1:
        print("Error: No hay suficientes datos para formar una ventana y un backtest.")
        return

    df_returns = df_raw.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    scaled_data = scaler.transform(df_returns.values)
    df_scaled = pd.DataFrame(scaled_data, index=df_returns.index, columns=df_returns.columns)

    last_price = df_raw.iloc[-1][TARGET_TICKER]
    last_date = df_raw.index[-1]
    
    client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
    write_api = client.write_api(write_options=SYNCHRONOUS)

    def execute_and_save_step(X_input, base_price, target_dt, is_backtest=False):
        preds_cls, preds_reg = mc_dropout_predict_mt(model, X_input, n_iter=100)
        
        mean_prob = float(np.mean(preds_cls))
        std_prob = float(np.std(preds_cls))
        mean_reg_scaled = float(np.mean(preds_reg))
        std_reg_scaled = float(np.std(preds_reg))
        
        # Desescalar Media
        dummy = np.zeros((1, df_returns.shape[1]))
        dummy[0, 0] = mean_reg_scaled
        expected_return = scaler.inverse_transform(dummy)[0, 0]
        expected_price = base_price * (1 + expected_return)
        
        # Bandas de Confianza (1 Varianza Estándar)
        dummy_upper = np.zeros((1, df_returns.shape[1]))
        dummy_upper[0, 0] = mean_reg_scaled + std_reg_scaled
        expected_return_upp = scaler.inverse_transform(dummy_upper)[0, 0]
        price_upper = base_price * (1 + expected_return_upp)
        
        dummy_lower = np.zeros((1, df_returns.shape[1]))
        dummy_lower[0, 0] = mean_reg_scaled - std_reg_scaled
        expected_return_low = scaler.inverse_transform(dummy_lower)[0, 0]
        price_lower = base_price * (1 + expected_return_low)

        # Solo emitir señales URGENTES si la incerteza es baja, logica para Dashboard
        signal = "WAIT"
        urgency = "LOW"
        if expected_return > 0.02:
            signal = "URGENT BUY"
            urgency = "HIGH"
        elif mean_prob > 0.55:
            signal = "BUY"
            urgency = "MEDIUM" if std_prob < 0.1 else "LOW"
        else:
            signal = "WAIT (Market Drop Expected)"
            urgency = "LOW"

        # Saltamos fines de semana si caen exacto, simple business day add (Opcional, pero Grafana interpola)
        if target_dt.weekday() >= 5: # Si es Sabado (5) o Domingo (6)
            target_dt += timedelta(days=2) if target_dt.weekday() == 5 else timedelta(days=1)

        record = {
            "measurement": "forecast_signal",
            "tags": {
                "ticker": TARGET_TICKER,
                "model_type": "CNN-LSTM_MultiTask",
                "signal": signal,
                "urgency": urgency,
                "is_backtest": str(is_backtest)
            },
            "time": target_dt.isoformat(),
            "fields": {
                "expected_price": float(expected_price),
                "expected_price_upper": float(price_upper),
                "expected_price_lower": float(price_lower),
                "expected_return_pct": float(expected_return * 100),
                "upward_probability": mean_prob,
                "probability_uncertainty": std_prob
            }
        }
        
        try:
            write_api.write(bucket=INFLUX_BUCKET_FORECAST, org=INFLUX_ORG, record=record)
            tag = "[BACKTEST]" if is_backtest else f"[FORECAST]"
            print(f" {tag} {target_dt.date()}: ${expected_price:.2f} (Banda: ${price_lower:.2f} - ${price_upper:.2f})")
        except Exception as e:
            print(f"Error guardando en InfluxDB: {e}")
            
        return expected_price, mean_reg_scaled

    print(f"\n💵 Último Precio Real (SI=F): ${last_price:.2f} el {last_date.date()}")
    
    print("\n🔙 1. PUNTO DE COMPARACIÓN (T-1 Predicts Today)")
    # La ventana de 10 días terminando el día anterior al actual
    X_backtest = df_scaled.iloc[-(WINDOW_SIZE + 1):-1].values.reshape(1, WINDOW_SIZE, df_scaled.shape[1])
    price_t_minus_1 = df_raw.iloc[-2][TARGET_TICKER] 
    execute_and_save_step(X_backtest, price_t_minus_1, last_date, is_backtest=True)

    print("\n🔮 2. PREDICCIÓN CON RECURSIVIDAD (Next 5 Days)")
    current_sequence = df_scaled.iloc[-WINDOW_SIZE:].values.reshape(1, WINDOW_SIZE, df_scaled.shape[1])
    current_iter_price = last_price
    
    # --- ANCLA VISUAL PARA GRAFANA (Conecta la línea verde con la amarilla en el día de hoy) ---
    anchor_record = {
        "measurement": "forecast_signal",
        "tags": {
            "ticker": TARGET_TICKER,
            "model_type": "CNN-LSTM_MultiTask",
            "signal": "ANCHOR",
            "urgency": "NONE",
            "is_backtest": "False"
        },
        "time": last_date.isoformat(),
        "fields": {
            "expected_price": float(last_price),
            "expected_price_upper": float(last_price),
            "expected_price_lower": float(last_price),
            "expected_return_pct": 0.0,
            "upward_probability": 0.5,
            "probability_uncertainty": 0.0
        }
    }
    write_api.write(bucket=INFLUX_BUCKET_FORECAST, org=INFLUX_ORG, record=anchor_record)
    print(f" ⚓ [ANCHOR] {last_date.date()}: ${last_price:.2f} (Punto inicial para Grafana)")

    # Predecir 5 pasos en el futuro
    for step in range(1, 6):
        forecast_dt = last_date + timedelta(days=step)
        
        predicted_price, pred_reg_scaled = execute_and_save_step(
            current_sequence, current_iter_price, forecast_dt, is_backtest=False
        )
        
        # Desplegar recursividad asumiendo que las demas variables (Oro, Dolar, etc) permanecen estables (Retorno 0 escalado)
        # Solo inyectamos el cambio predecido de Plata para la siguiente iteracion
        zero_input = np.zeros((1, df_returns.shape[1]))
        zero_scaled = scaler.transform(zero_input)[0] 
        zero_scaled[0] = pred_reg_scaled # SI=F es index 0
        
        next_row = zero_scaled.reshape(1, 1, df_scaled.shape[1])
        
        # Deslizar ventana: Botar el 1ero, meter el nuevo al final
        current_sequence = np.append(current_sequence[:, 1:, :], next_row, axis=1)
        current_iter_price = predicted_price

    print("\n✅ ¡Cadena Markoviana de 5 días e Intervalos insertados en la Base de Datos!")
    client.close()

if __name__ == "__main__":
    run_prediction()
