import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

st.set_page_config(page_title="Predicción Hipotecaria TFM", layout="centered")

st.title("🏠 Predicción de Ejecuciones Hipotecarias")

# --- FUNCIÓN DE CARGA ROBUSTA (SOLUCIÓN DE RUTAS) ---
@st.cache_resource
def load_resources():
    # 1. Obtener la ruta absoluta de este archivo (app.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 2. Subir un nivel para llegar a la raíz del proyecto (donde están 'models' y 'data')
    root_path = os.path.dirname(current_dir)
    
    # 3. Construir las rutas exactas
    model_path = os.path.join(root_path, 'models', 'modelo_ejecuciones.joblib')
    data_path = os.path.join(root_path, 'data', 'datos_historicos.csv')
    
    # (Opcional) Imprimir rutas en los logs del servidor para depurar
    print(f"Buscando modelo en: {model_path}")
    print(f"Buscando datos en: {data_path}")

    # 4. Comprobación de seguridad
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        # Si falla, devolvemos las rutas para que sepas dónde está buscando
        return None, None, model_path, data_path

    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        return model, df, model_path, data_path
    except Exception as e:
        st.error(f"Error leyendo archivos: {e}")
        return None, None, model_path, data_path

# Ejecutar la carga
model, df_clean, m_path, d_path = load_resources()

if model is not None:
    st.success("✅ Modelo y datos cargados correctamente.")
    
    # --- TU LÓGICA DE LA APP ---
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
        
        # Obtener último valor real disponible para usar como 'lag'
        if not df_clean[mask].empty:
            val_real = df_clean[mask]['total'].iloc[-1]
        else:
            val_real = 0
        
        # Crear DataFrame de entrada con las columnas que espera el modelo
        input_data = pd.DataFrame({
            'periodo': [anio], 
            'lag_1': [val_real], 
            'lag_2': [val_real], 
            'rolling_mean_2': [val_real],
            'com_nom': [comunidad], 
            'titular': [titular]
        })
        
        try:
            pred = model.predict(input_data)[0]
            st.metric(f"Predicción {anio}", int(pred))
        except Exception as e:
            st.error(f"Error en la predicción: {e}")
            st.write("Verifica que las columnas del Excel coinciden con las del entrenamiento.")

else:
    #Mensaje de error detallado si falla la carga
    st.error("⚠️ Faltan archivos.")
    st.warning(f"La app está buscando el modelo aquí:\n{m_path}")
    st.warning(f"Y los datos aquí:\n{d_path}")
    st.info("Asegúrate de que las carpetas 'models' y 'data' están en la raíz del repositorio en GitHub.")

if model is not None:
    st.success("✅ Modelo y datos cargados correctamente.")
    
    #Selectores
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
