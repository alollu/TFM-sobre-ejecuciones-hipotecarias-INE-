import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Predicción Hipotecaria TFM", layout="centered")

st.title("🏠 Predicción de Ejecuciones Hipotecarias")

# Cargar recursos
@st.cache_resource
def load_resources():
    model_path = 'models/modelo_ejecuciones.joblib'
    data_path = 'data/datos_historicos.csv'
    
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        return None, None

    model = joblib.load(model_path)
    df = pd.read_csv(data_path)
    return model, df

model, df_clean = load_resources()

if model is not None:
    st.success("✅ Modelo y datos cargados correctamente.")
    
    # Selectores
    comunidades = sorted(df_clean['com_nom'].unique())
    titulares = sorted(df_clean['titular'].unique())
    
    st.sidebar.header("Configuración")
    comunidad = st.sidebar.selectbox("Comunidad", comunidades)
    titular = st.sidebar.selectbox("Titular", titulares)
    anio = st.sidebar.slider("Año", 2025, 2030, 2025)
    
    if st.sidebar.button("Predecir"):
        # Lógica simplificada para demo
        mask = (df_clean['com_nom'] == comunidad) & (df_clean['titular'] == titular)
        val_real = df_clean[mask]['total'].iloc[-1] if not df_clean[mask].empty else 0
        
        input_data = pd.DataFrame({
            'periodo': [anio], 'com_nom': [comunidad], 'titular': [titular],
            'lag_1': [val_real], 'lag_2': [val_real], 'rolling_mean_2': [val_real]
        })
        
        pred = model.predict(input_data)[0]
        st.metric(f"Predicción {anio}", int(pred))
else:
    st.error("⚠️ Faltan archivos en 'models/' o 'data/'.")
