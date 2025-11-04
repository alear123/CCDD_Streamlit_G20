# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import altair as alt
import os
from sklearn.base import BaseEstimator, TransformerMixin

st.set_page_config(layout="wide", page_title="Predicción de demanda por región")

# ---------------------------
# CONFIGURACIÓN DE REGIONES
# ---------------------------
REGION_COORDS = {
    "edelap": {"lat": -34.921, "lon": -57.954},  # La Plata
    "edesur": {"lat": -34.615, "lon": -58.425},  # CABA sur
    "edenor": {"lat": -34.567, "lon": -58.447}   # CABA norte
}

MODEL_FOLDER = "models"




class FeatureEngineerTemporal(BaseEstimator, TransformerMixin):
    """
    Transformer de Scikit-learn que divide la columna fecha en hora, dia_semana, mes,.

    """

    def __init__(self, drop_original_fecha=True):
        self.drop_original_fecha = drop_original_fecha

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()

        # Asegurar columna fecha
        if 'fecha' not in df.columns:
            raise ValueError("No se encontró la columna 'fecha' en el DataFrame.")

        # Asegurar datetime
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

        # Features de calendario
        df['hora'] = df['fecha'].dt.hour
        df['dia_semana'] = df['fecha'].dt.weekday
        df['mes'] = df['fecha'].dt.month

        # Codificación cíclica
        df['hora_sin'] = np.sin(2 * np.pi * df['hora'] / 24)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora'] / 24)
        df['mes_sin'] = np.sin(2 * np.pi * df['mes'] / 12)
        df['mes_cos'] = np.cos(2 * np.pi * df['mes'] / 12)

        # Dropear fecha original si corresponde
        if self.drop_original_fecha:
            df = df.drop(columns=['fecha'], errors='ignore')

        return df

# ---------------------------
# FUNCIÓN: CARGAR MODELO
# ---------------------------
def load_model(region, model_folder=MODEL_FOLDER):
    """Carga el modelo preentrenado desde models/model_<region>.pkl"""
    model_path = os.path.join(model_folder, f"model_{region}.pkl")

    if not os.path.exists(model_path):
        st.error(f"No se encontró el modelo para la región '{region}' en {model_path}")
        return None

    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo {model_path}: {e}")
        return None

# ---------------------------
# FUNCIÓN: OBTENER PRONÓSTICO OPEN-METEO
# ---------------------------
def fetch_open_meteo_forecast(lat, lon, timezone="America/Argentina/Buenos_Aires", forecast_days=7):
    """Obtiene pronóstico meteorológico horario (7 días) desde Open-Meteo"""
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
    hourly = data.get("hourly", {})
    df = pd.DataFrame(hourly)
    df["fecha"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])
    df = df.rename(columns={
        "temperature_2m": "temperature_2m",
        "relativehumidity_2m": "relative_humidity_2m",
        "precipitation": "precipitation",
        "cloudcover": "cloudcover",
        "pressure_msl": "pressure_msl",
        "windspeed_10m": "wind_speed_10m",
        "winddirection_10m": "wind_direction_10m"
    })
    df["fin_de_semana"] = df["fecha"].dt.weekday.isin([5, 6]).astype(int)
    return df

# ---------------------------
# FUNCIÓN: PREPARAR FEATURES PARA PREDICCIÓN
# ---------------------------
def prepare_features(df):
    df["hora"] = df["fecha"].dt.hour
    df["mes"] = df["fecha"].dt.month
    df["hora_sin"] = np.sin(2 * np.pi * df["hora"] / 24)
    df["hora_cos"] = np.cos(2 * np.pi * df["hora"] / 24)
    df["mes_sin"] = np.sin(2 * np.pi * df["mes"] / 12)
    df["mes_cos"] = np.cos(2 * np.pi * df["mes"] / 12)

    features = [
        "temperature_2m", "relative_humidity_2m", "precipitation", "cloudcover",
        "pressure_msl", "wind_speed_10m", "wind_direction_10m",
        "hora_sin", "hora_cos", "mes_sin", "mes_cos", "fin_de_semana"
    ]

    for f in features:
        if f not in df.columns:
            df[f] = 0.0

    return df, df[features].fillna(0)

# ---------------------------
# INTERFAZ STREAMLIT
# ---------------------------
st.title("Predicción de Demanda Eléctrica (7 días)")
st.markdown("Selecciona la región para obtener el pronóstico del clima y la predicción de demanda correspondiente.")

region = st.selectbox("Región:", list(REGION_COORDS.keys()))

model = load_model(region)
if model is None:
    st.stop()

coords = REGION_COORDS[region]
st.info(f"Coordenadas de {region}: {coords['lat']}, {coords['lon']}")

# Obtener pronóstico
with st.spinner("Obteniendo pronóstico meteorológico..."):
    df_forecast = fetch_open_meteo_forecast(coords["lat"], coords["lon"])

if df_forecast.empty:
    st.error("No se pudo obtener el pronóstico.")
    st.stop()

columnas_necesarias = [
    "fecha", "temperature_2m", "relative_humidity_2m", "precipitation",
    "cloudcover", "pressure_msl", "wind_speed_10m", "wind_direction_10m",
    "region", "fin_de_semana", "estacion"
]

for col in columnas_necesarias:
    if col not in df_forecast.columns:
        if col == "region":
            df_forecast[col] = region  # usa la región seleccionada en la app
        elif col == "estacion":
            # podés estimar estación según mes
            df_forecast["mes"] = df_forecast["fecha"].dt.month
            df_forecast["estacion"] = df_forecast["mes"].map({
                12: "verano", 1: "verano", 2: "verano",
                3: "otoño", 4: "otoño", 5: "otoño",
                6: "invierno", 7: "invierno", 8: "invierno",
                9: "primavera", 10: "primavera", 11: "primavera"
            })
        else:
            df_forecast[col] = 0
# Preparar features y predecir
df_forecast["pred_dem"] = model.predict(df_forecast)

# ---------------------------
# VISUALIZACIONES
# ---------------------------
st.subheader("Predicción horaria de demanda")
chart1 = alt.Chart(df_forecast).mark_line().encode(
    x=alt.X("fecha:T", title="Fecha"),
    y=alt.Y("pred_dem:Q", title="Demanda predicha"),
    tooltip=["fecha", "pred_dem"]
).interactive()
st.altair_chart(chart1, use_container_width=True)

st.subheader(" Temperatura vs Demanda")
chart2 = alt.layer(
    alt.Chart(df_forecast).mark_line(color="orange").encode(
        x="fecha:T", y=alt.Y("temperature_2m:Q", title="Temperatura (°C)")
    ),
    alt.Chart(df_forecast).mark_line(color="blue").encode(
        x="fecha:T", y=alt.Y("pred_dem:Q", title="Demanda predicha")
    )
).resolve_scale(y="independent").interactive()
st.altair_chart(chart2, use_container_width=True)

st.subheader("Importancia de características (si el modelo lo permite)")
try:
    estimator = model.named_steps.get("model", model)
    if hasattr(estimator, "feature_importances_"):
        # Intentar obtener los nombres reales desde el preprocesador
        try:
            preprocessor = model.named_steps.get("preprocessing", None)
            if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
                feature_names = preprocessor.get_feature_names_out()
            else:
                feature_names = [f"Feature {i}" for i in range(len(estimator.feature_importances_))]
        except Exception:
            feature_names = [f"Feature {i}" for i in range(len(estimator.feature_importances_))]

        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": estimator.feature_importances_
        }).sort_values("importance", ascending=False)

        chart3 = alt.Chart(fi).mark_bar().encode(
            x="importance:Q",
            y=alt.Y("feature:N", sort="-x"),
            tooltip=["feature", "importance"]
        )
        st.altair_chart(chart3, use_container_width=True)
    else:
        st.info("El modelo no expone importancias de características.")
except Exception as e:
    st.warning(f"No se pudo mostrar importancias: {e}")


# ---------------------------
# DESCARGA
# ---------------------------
csv = df_forecast.to_csv(index=False)
st.download_button("Descargar predicciones (CSV)", csv, file_name=f"predicciones_{region}.csv", mime="text/csv")

st.success("Predicción completada correctamente.")
