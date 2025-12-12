import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

# 1. Cargar datos
print("Cargando datos...")
try:
    df = pd.read_csv('../data/datos_historicos.csv')
except:
    df = pd.read_csv('data/datos_historicos.csv')

# 2. Configurar el entrenamiento
print("Entrenando modelo compatible...")
features_num = ['periodo', 'lag_1', 'lag_2', 'rolling_mean_2']
features_cat = ['com_nom', 'titular']

X = df[features_num + features_cat]
y = df['total']

preprocess = ColumnTransformer(transformers=[
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), features_cat),
    ('num', StandardScaler(), features_num)
])

# Usamos LinearRegression que es ligero y robusto
modelo = Pipeline([('prep', preprocess), ('model', LinearRegression())])
modelo.fit(X, y)

# 3. Guardar el modelo nuevo
print("Guardando archivo...")
joblib.dump(modelo, 'modelo_ejecuciones.joblib')
print("¡ÉXITO! Modelo regenerado.")