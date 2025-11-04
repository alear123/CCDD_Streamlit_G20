# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import altair as alt
import os
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(layout="wide", page_title="Predicción de demanda eléctrica", page_icon="⚡")

# ---------------------------
# CONFIGURACIÓN DE REGIONES
# ---------------------------
REGION_COORDS = {
    "edelap": {"lat": -34.921, "lon": -57.954},  # La Plata
    "edesur": {"lat": -34.615, "lon": -58.425},  # CABA sur
    "edenor": {"lat": -34.567, "lon": -58.447}   # CABA norte
}
MODEL_FOLDER = "models"

# ---------------------------
# FEATURE ENGINEER
# ---------------------------
class FeatureEngineerTemporal(BaseEstimator, TransformerMixin):
    def __init__(self, drop_original_fecha=True):
        self.drop_original_fecha = drop_original_fecha

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        if 'fecha' not in df.columns:
            raise ValueError("No se encontró la columna 'fecha'.")
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['hora'] = df['fecha'].dt.hour
        df['dia_semana'] = df['fecha'].dt.weekday
        df['mes'] = df['fecha'].dt.month
        df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)
        if self.drop_original_fecha:
            df = df.drop(columns=['fecha'], errors='ignore')
        return df

# ---------------------------
# FUNCIONES AUXILIARES
# ---------------------------
def load_model(region, model_folder=MODEL_FOLDER):
    model_path = os.path.join(model_folder, f"model_{region}.pkl")
    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo para la región '{region}' en {model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

def fetch_open_meteo_forecast(lat, lon, timezone="America/Argentina/Buenos_Aires", forecast_days=7):
    base = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,relativehumidity_2m,precipitation,cloudcover,pressure_msl,windspeed_10m,winddirection_10m",
        "forecast_days": forecast_days,
        "timezone": timezone
    }
    response = requests.get(base, params=params)
    response.raise_for_status()
    data = response.json()
    df = pd.DataFrame(data["hourly"])
    df["fecha"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df["fin_de_semana"] = df["fecha"].dt.weekday.isin([5,6]).astype(int)
    df = df.rename(columns={
        "temperature_2m": "temperature_2m",
        "relativehumidity_2m": "relative_humidity_2m",
        "precipitation": "precipitation",
        "cloudcover": "cloudcover",
        "pressure_msl": "pressure_msl",
        "windspeed_10m": "wind_speed_10m",
        "winddirection_10m": "wind_direction_10m"
    })
    return df

def align_forecast(df_forecast, region_name):
    df = df_forecast.copy()
    df['region'] = region_name
    df['estacion'] = df['fecha'].dt.month.map({
        12:"verano",1:"verano",2:"verano",
        3:"otoño",4:"otoño",5:"otoño",
        6:"invierno",7:"invierno",8:"invierno",
        9:"primavera",10:"primavera",11:"primavera"
    })
    expected_cols = [
        "fecha","cloudcover","pressure_msl","precipitation","temperature_2m",
        "wind_speed_10m","wind_direction_10m","relative_humidity_2m",
        "region","fin_de_semana","estacion"
    ]
    for c in expected_cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[expected_cols]

# ---------------------------
# SIDEBAR
# ---------------------------
# ---------------------------
# TEXTO DE INTRODUCCIÓN
# ---------------------------
st.markdown(
    """
    # ⚡ Predicción de Demanda Eléctrica por Región

    Bienvenido a la herramienta de predicción de demanda eléctrica.  
    Esta aplicación permite:
    
    - Obtener el pronóstico horario de demanda eléctrica para las principales regiones.
    - Visualizar la relación entre temperatura y demanda.
    - Consultar la importancia de las variables que influyen en la predicción.
    - Descargar los resultados para análisis posterior.

    **Cómo usarla:**
    1. Selecciona la región y los días a predecir en la barra lateral.
    2. Visualiza los gráficos y la tabla de predicciones.
    3. Descarga los resultados si lo deseas.

    💡 Esta herramienta utiliza modelos de aprendizaje automático entrenados con datos históricos y pronósticos meteorológicos.
    """
)

# ---------------------------
# SIDEBAR
# ---------------------------

st.sidebar.title("Configuración")
region = st.sidebar.selectbox("Selecciona la región:", list(REGION_COORDS.keys()))
forecast_days = st.sidebar.slider("Días a predecir:", 1, 14, 7)
model = load_model(region)
if model is None:
    st.stop()

# ---------------------------
# OBTENER DATOS
# ---------------------------
coords = REGION_COORDS[region]
with st.spinner("Obteniendo pronóstico meteorológico..."):
    df_forecast = fetch_open_meteo_forecast(coords["lat"], coords["lon"], forecast_days=forecast_days)

df_forecast_aligned = align_forecast(df_forecast, region)

with st.spinner("Generando predicciones..."):
    df_forecast["pred_dem"] = model.predict(df_forecast_aligned)

# ---------------------------
# TARJETAS DE ESTADÍSTICAS
# ---------------------------
st.subheader(f"Resumen de la demanda para '{region}'")
col1, col2, col3 = st.columns(3)
col1.metric("Máx. demanda", f"{df_forecast['pred_dem'].max():.2f} MW")
col2.metric("Mín. demanda", f"{df_forecast['pred_dem'].min():.2f} MW")
col3.metric("Demanda promedio", f"{df_forecast['pred_dem'].mean():.2f} MW")

# ---------------------------
# VISUALIZACIONES
# ---------------------------
st.subheader("Predicción horaria de demanda")
chart1 = alt.Chart(df_forecast).mark_line(color="blue").encode(
    x="fecha:T",
    y="pred_dem:Q",
    tooltip=["fecha","pred_dem"]
).interactive()
st.altair_chart(chart1, use_container_width=True)

st.subheader("Temperatura vs Demanda")
chart2 = alt.layer(
    alt.Chart(df_forecast).mark_line(color="orange").encode(
        x="fecha:T", y="temperature_2m:Q", tooltip=["fecha","temperature_2m"]
    ),
    alt.Chart(df_forecast).mark_line(color="blue").encode(
        x="fecha:T", y="pred_dem:Q", tooltip=["fecha","pred_dem"]
    )
).resolve_scale(y="independent").interactive()
st.altair_chart(chart2, use_container_width=True)

# ---------------------------
# IMPORTANCIA DE FEATURES
# ---------------------------
st.subheader(f"Importancia de características para '{region}'")
try:
    estimator = model.named_steps.get("model", model)
    if hasattr(estimator,"feature_importances_"):
        importances = estimator.feature_importances_
        feature_names = None
        if hasattr(model, "named_steps") and "preprocessing" in model.named_steps:
            preprocessor = model.named_steps["preprocessing"]
            if hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
                feature_names = [name.split("__")[-1] for name in feature_names]
        if feature_names is None:
            feature_names = df_forecast_aligned.drop(columns=["fecha"]).columns[:len(importances)]
        fi = pd.DataFrame({"feature": feature_names[:len(importances)], "importance": importances})\
                .sort_values("importance", ascending=False)
        with st.expander("Ver tabla de importancia"):
            st.dataframe(fi.head(10))
        chart3 = alt.Chart(fi.head(10)).mark_bar().encode(
            x="importance:Q", y=alt.Y("feature:N", sort="-x"),
            color="importance:Q", tooltip=["feature","importance"]
        ).properties(height=350)
        st.altair_chart(chart3,use_container_width=True)
    else:
        st.info(f"El modelo para '{region}' no tiene 'feature_importances_'.")
except Exception as e:
    st.warning(f"No se pudo mostrar importancias: {e}")

# ---------------------------
# DESCARGA
# ---------------------------
csv = df_forecast.to_csv(index=False)
st.download_button(" Descargar predicciones (CSV)", csv,
                   file_name=f"predicciones_{region}.csv", mime="text/csv")
st.success(" Predicción completada correctamente.")
