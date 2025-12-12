import pandas as pd
import joblib
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# --- 1. CONFIGURACIÓN DE RUTAS ---
base_dir = os.getcwd()
data_path = os.path.join(base_dir, 'data', 'datos_historicos.csv')
models_dir = os.path.join(base_dir, 'models')
model_save_path = os.path.join(models_dir, 'modelo_ejecuciones.joblib')

print(f"📍 Buscando datos en: {data_path}")

# Crear carpeta models si no existe
os.makedirs(models_dir, exist_ok=True)

# --- 2. CARGAR Y PREPARAR DATOS ---
if not os.path.exists(data_path):
    print("❌ ERROR: No encuentro el archivo data/datos_historicos.csv")
    exit()

df = pd.read_csv(data_path, sep=None, engine='python') # sep=None autodetecta ; o ,

# Limpieza básica de nombres
df.columns = [c.lower().strip() for c in df.columns]

# --- 3. CREAR VARIABLES (LAGS) SI NO EXISTEN ---
# Esto soluciona el error KeyError: 'lag_1'
if 'lag_1' not in df.columns:
    print("⚙️  Generando variables matemáticas (Lags)...")
    # Aseguramos que está ordenado
    df = df.sort_values(['com_nom', 'titular', 'periodo'])
    
    # Creamos las columnas necesarias para el modelo
    df['lag_1'] = df.groupby(['com_nom', 'titular'])['total'].shift(1)
    df['lag_2'] = df.groupby(['com_nom', 'titular'])['total'].shift(2)
    df['rolling_mean_2'] = df.groupby(['com_nom', 'titular'])['total'].transform(
        lambda x: x.shift(1).rolling(window=2).mean()
    )
    # Quitamos los huecos vacíos (NaN) que se crean al desplazar
    df = df.dropna(subset=['lag_1', 'lag_2', 'rolling_mean_2'])
    
    # Guardamos este CSV mejorado para que la App lo use
    df.to_csv(data_path, index=False)
    print("✅ Datos procesados y actualizados.")

# --- 4. ENTRENAR MODELO ---
print("🚀 Entrenando modelo (LinearRegression)...")

features_num = ['periodo', 'lag_1', 'lag_2', 'rolling_mean_2']
features_cat = ['com_nom', 'titular']

X = df[features_num + features_cat]
y = df['total']

preprocess = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat),
    ('num', StandardScaler(), features_num)
])

model = Pipeline([('prep', preprocess), ('model', LinearRegression())])
model.fit(X, y)

# --- 5. GUARDAR ---
print(f"💾 Guardando modelo en: {model_save_path}")
joblib.dump(model, model_save_path)
print("✅ ¡ÉXITO! Todo listo para subir.")
