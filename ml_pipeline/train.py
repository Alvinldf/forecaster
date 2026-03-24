import os
import warnings
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.tensorflow
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Input, Conv1D, LSTM, Dense, Dropout
from influxdb_client import InfluxDBClient

# 1. Configuración global
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

try:
    from config import INFLUX_URL, INFLUX_TOKEN, INFLUX_ORG, INFLUX_BUCKET
except ImportError:
    INFLUX_URL = "http://localhost:8086"
    INFLUX_TOKEN = "your-token"
    INFLUX_ORG = "your-org"
    INFLUX_BUCKET = "MarketData"

TARGET_TICKER = 'SI=F'
RELEVANT_TICKERS = ['SLV', 'NG=F', 'SIL', 'CL=F']
ALL_TICKERS = [TARGET_TICKER] + RELEVANT_TICKERS

# MLflow Config
MLFLOW_URI = "http://localhost:5000"
mlflow.set_tracking_uri(MLFLOW_URI)
mlflow.set_experiment("Silver_Production_Models")

def fetch_multivariate_data(tickers, days=1825): # 5 años
    print("📥 Recuperando datos desde InfluxDB...")
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

def run_training_pipeline():
    print("🚀 Iniciando Pipeline de Entrenamiento para Producción...")
    os.makedirs('models', exist_ok=True)
    
    # 1. INGESTA
    df_raw = fetch_multivariate_data(ALL_TICKERS)
    df_returns = df_raw.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
    
    # 2. ESCALAMIENTO Y SERIALIZACIÓN DEL SCALER
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_returns.values)
    df_scaled = pd.DataFrame(scaled_data, index=df_returns.index, columns=df_returns.columns)
    
    scaler_path = "models/scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler serializado exitosamente en {scaler_path}")
    
    # 3. PREPARACIÓN DE SECUENCIAS
    window_size = 10
    def create_sequences(data):
        X, y = [], []
        for i in range(len(data) - window_size):
            X.append(data.iloc[i : (i + window_size)].values)
            y.append(data.iloc[i + window_size][TARGET_TICKER])
        return np.array(X), np.array(y)
        
    train_size = int(len(df_scaled) * 0.8)
    val_size = int(len(df_scaled) * 0.1)
    
    train_df = df_scaled.iloc[:train_size]
    val_df = df_scaled.iloc[train_size:train_size + val_size]
    
    X_train, y_train_reg = create_sequences(train_df)
    X_val, y_val_reg = create_sequences(val_df)
    
    zero_input = np.zeros((1, df_returns.shape[1]))
    zero_return_scaled = scaler.transform(zero_input)[0, 0]
    
    y_train_class = (y_train_reg > zero_return_scaled).astype(int)
    y_val_class = (y_val_reg > zero_return_scaled).astype(int)
    
    # Pesos balanceados para Clasificación
    weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_class), y=y_train_class)
    class_weights = dict(zip(np.unique(y_train_class), weights))
    
    n_features = X_train.shape[2]
    
    # =========================================================
    # MODELO 1: Regresión CNN-LSTM
    # =========================================================
    print("\n🧠 Entrenando Modelo 1: Regresión CNN-LSTM...")
    with mlflow.start_run(run_name="Prod_Regresion_CNN_LSTM"):
        model_reg = Sequential([
            InputLayer(shape=(window_size, n_features)),
            Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'),
            Dropout(0.2),
            LSTM(64, return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        model_reg.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse')
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
        model_reg.fit(X_train, y_train_reg, validation_data=(X_val, y_val_reg), epochs=100, batch_size=32, verbose=0, callbacks=[early_stop])
        
        reg_path = "models/cnn_lstm_regressor.keras"
        model_reg.save(reg_path)
        mlflow.log_artifact(reg_path)
        mlflow.log_artifact(scaler_path)
        print(f"✅ Regresión guardada en {reg_path} y registrada en MLflow")

    # =========================================================
    # MODELO 2: Multi-Task CNN-LSTM
    # =========================================================
    print("\n🧠 Entrenando Modelo 2: Multi-Task CNN-LSTM...")
    with mlflow.start_run(run_name="Prod_MultiTask_CNN_LSTM"):
        inputs = Input(shape=(window_size, n_features))
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
        x = Dropout(0.2)(x)
        x = LSTM(64, return_sequences=False)(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        
        out_cls = Dense(1, activation='sigmoid', name='clasificador')(x)
        out_reg = Dense(1, activation='linear', name='magnitud')(x)
        model_mt = Model(inputs=inputs, outputs={'clasificador': out_cls, 'magnitud': out_reg})
        
        model_mt.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
            loss={'clasificador': 'binary_crossentropy', 'magnitud': 'mse'},
            loss_weights={'clasificador': 1.0, 'magnitud': 0.5}
        )
        
        early_stop_mt = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        sample_weights_cls = np.array([class_weights[y] for y in y_train_class])
        sample_weights_reg = np.ones_like(y_train_reg)
        
        model_mt.fit(
            X_train, 
            {'clasificador': y_train_class, 'magnitud': y_train_reg},
            validation_data=(X_val, {'clasificador': y_val_class, 'magnitud': y_val_reg}),
            epochs=150, batch_size=32, verbose=0, callbacks=[early_stop_mt],
            sample_weight={'clasificador': sample_weights_cls, 'magnitud': sample_weights_reg}
        )
        
        mt_path = "models/cnn_lstm_multitask.keras"
        model_mt.save(mt_path)
        mlflow.log_artifact(mt_path)
        mlflow.log_artifact(scaler_path)
        print(f"✅ Multi-Task guardado en {mt_path} y registrado en MLflow")

    print("\n🎉 Los modelos y scalers están listos para Inferencia (predict.py).")

if __name__ == "__main__":
    run_training_pipeline()