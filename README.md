# Análisis y Predicción de Ejecuciones Hipotecarias en España
Este proyecto de Data Science analiza la evolución de las ejecuciones hipotecarias en España (datos INE) y desarrolla un modelo predictivo basado en Machine Learning para anticipar tendencias en el periodo 2025-2027.

Trabajo de Fin de Máster (TFM) - Road to Data Science.

![Python](https://img.shields.io/badge/python-3.10-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-draft-orange)


## 🎯 Objetivo
Identificar patrones temporales y geográficos en las ejecuciones hipotecarias y proporcionar una herramienta predictiva que ayude a anticipar volúmenes futuros por Comunidad Autónoma y tipo de titular.

## 🗂️ Estructura del Proyecto
* **`data/`**: Contiene los datasets históricos (procesados y raw).
* **`notebooks/`**:
    * `01_Analisis_EDA.ipynb`: Limpieza de datos y análisis exploratorio.
    * `02_Modelado_Prediccion.ipynb`: Ingeniería de variables, entrenamiento (XGBoost) y validación.
* **`models/`**: Archivo binario del modelo entrenado (`modelo_ejecuciones.joblib`).
* **`src/`**: Código fuente de la aplicación de despliegue.

## 🛠️ Tecnologías Utilizadas
* **Lenguaje:** Python 3.9+
* **Manipulación de Datos:** Pandas, Numpy.
* **Visualización:** Matplotlib, Seaborn.
* **Machine Learning:** XGBoost (Modelo seleccionado), Scikit-learn.
* **Forecasting:** Prophet, Series Temporales.
* **Despliegue:** Streamlit.

## 📊 Resultados Destacados
* El modelo basado en **Gradient Boosting (XGBoost)** superó a los modelos lineales y ARIMA, logrando una mayor precisión gracias a la incorporación de variables de retardo (*lags*).
* Se observa una **tendencia de estabilización** para el periodo 2025-2027, aunque regiones como Cataluña, Comunidad Valenciana y Andalucía mantienen los mayores volúmenes absolutos.

## 🚀 Cómo ejecutar la App (Streamlit)
Para visualizar las predicciones de forma interactiva:

1. Instalar dependencias:
   ```bash
   pip install -r requirements.txt
