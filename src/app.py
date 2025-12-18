#importo Librerías
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import os

# CONFIGURACIÓN DE PÁGINA 
st.set_page_config(page_title="Predicción Hipotecaria TFM", layout="centered")
st.title("🏠 Predicción de Ejecuciones Hipotecarias")

# FUNCIÓN DE CARGA DE DATOS
@st.cache_resource
def load_resources():
    #1.Obtenemos la ruta donde está este archivo (src/app.py)
    ruta_actual = os.path.dirname(os.path.abspath(__file__))
    # Subimos un nivel para llegar a la raíz del proyecto
    ruta_raiz = os.path.dirname(ruta_actual)
    
    #2.Construimos las rutas completas a los archivos
    model_path = os.path.join(ruta_raiz, 'models', 'modelo_ejecuciones.joblib')
    data_path = os.path.join(ruta_raiz, 'data', 'datos_historicos.csv')
    
    # para ver en los logs dónde está buscando
    print(f"Buscando modelo en: {model_path}")

    #3.Verificamos si existen antes de cargar
    if not os.path.exists(model_path) or not os.path.exists(data_path):
        # Devolvemos None y las rutas para mostrar el error
        return None, None, model_path, data_path

    #4.Cargamos el modelo y los datos
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(data_path)
        return model, df, model_path, data_path
    except Exception as e:
        st.error(f"Error técnico leyendo archivos: {e}")
        return None, None, model_path, data_path

#Ejecutamos la carga
model, df_clean, ruta_modelo, ruta_datos = load_resources()

#  INTERFAZ DE LA APLICACIÓN 
if model is not None and df_clean is not None:
    st.success("✅ Sistema cargado y listo.")
    
    # Panel lateral de configuración
    st.sidebar.header("Parámetros de Predicción")
    
    # Selectores automáticos basados en los datos
    comunidades = sorted(df_clean['com_nom'].unique())
    titulares = sorted(df_clean['titular'].unique())
    
    comunidad = st.sidebar.selectbox("Comunidad Autónoma", comunidades)
    titular = st.sidebar.selectbox("Tipo de Titular", titulares)
    anio = st.sidebar.slider("Año a predecir", 2025, 2030, 2025)
    
    # Botón de predicción
    if st.sidebar.button("Calcular Predicción", type="primary"):
        # Buscamos el último dato real para usarlo como base (Lags)
        mask = (df_clean['com_nom'] == comunidad) & (df_clean['titular'] == titular)
        
        if not df_clean[mask].empty:
            # Cogemos el último valor conocido de 'total'
            val_real = df_clean[mask].sort_values('periodo')['total'].iloc[-1]
        else:
            val_real = 0 # Valor por defecto si no hay datos
        
        # Preparamos los datos para el modelo (mismo formato que en el entrenamiento)
        input_data = pd.DataFrame({
            'periodo': [anio], 
            'com_nom': [comunidad], 
            'titular': [titular],
            'lag_1': [val_real],        # Asumimos inercia del último año
            'lag_2': [val_real],        # Simplificación para la demo
            'rolling_mean_2': [val_real]
        })
        
        try:
            #Hacemos la predicción
            prediccion = model.predict(input_data)[0]
            
            #Mostramos el resultado
            st.divider()
            st.subheader(f"Resultados para {comunidad} ({anio})")
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric(label="Ejecuciones Previstas", value=f"{int(prediccion)}")
            with col2:
                st.info(f"Basado en último dato real: {int(val_real)}")
                
        except Exception as e:
            st.error(f"Error al generar la predicción: {e}")
            st.write("Detalles para depuración:", input_data)

else:
    # PANTALLA DE ERROR SI NO ENCUENTRA LOS ARCHIVOS 
    st.error("⚠️ Error Crítico: No se encuentran los archivos de datos.")
    st.warning("El sistema está buscando en estas rutas:")
    st.code(f"Modelo: {ruta_modelo}")
    st.code(f"Datos: {ruta_datos}")
    st.info("Por favor, verifica que las carpetas 'models' y 'data' están en la raíz de tu GitHub.")
