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

# ---------------------------
# Alinear df_forecast con las columnas raw que espera el preprocessor
# ---------------------------
def align_forecast_with_preprocessor(df_forecast, model, region_name=None):
    """
    Alinea df_forecast con las columnas que el preprocessor esperaba en entrenamiento.
    - Añade columnas faltantes con valores por defecto.
    - Reordena las columnas al mismo orden.
    - Forza tipos básicos (numerics -> float, categoricals -> object).
    """
    df = df_forecast.copy()
    # asegurar columna fecha como datetime
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')

    preprocessor = None
    if hasattr(model, "named_steps") and "preprocessing" in model.named_steps:
        preprocessor = model.named_steps["preprocessing"]

    if preprocessor is None:
        # nada que alinear si no hay preprocessor
        return df

    # 1) columnas raw que esperaba el preprocessor en el fit (si existe)
    expected_raw = getattr(preprocessor, "feature_names_in_", None)
    if expected_raw is None:
        # fallback: intentar obtener de ColumnTransformer si existe attribute input_features
        try:
            expected_raw = preprocessor.steps[0][1].feature_names_in_
        except Exception:
            expected_raw = None

    if expected_raw is None:
        # no hay información; devolvemos df tal cual
        return df

    # 2) añadir columnas faltantes con valores por defecto razonables
    for col in expected_raw:
        if col not in df.columns:
            if col == "region":
                df[col] = region_name if region_name is not None else "unknown"
            elif col in ["estacion", "dia_semana", "mes", "hora", "fin_de_semana"]:
                # si depende de fecha, intentar derivar
                if 'fecha' in df.columns:
                    if col == "estacion":
                        df['mes'] = df['fecha'].dt.month
                        df[col] = df['mes'].map({
                            12: "verano",1:"verano",2:"verano",
                            3:"otoño",4:"otoño",5:"otoño",
                            6:"invierno",7:"invierno",8:"invierno",
                            9:"primavera",10:"primavera",11:"primavera"
                        })
                    elif col == "dia_semana":
                        df[col] = df['fecha'].dt.weekday
                    elif col == "hora":
                        df[col] = df['fecha'].dt.hour
                    elif col == "mes":
                        df[col] = df['fecha'].dt.month
                    elif col == "fin_de_semana":
                        df[col] = df['fecha'].dt.weekday.isin([5,6]).astype(int)
                else:
                    # fallback
                    df[col] = 0
            else:
                # si parece numérica -> 0.0, sino -> 'missing'
                df[col] = 0.0

    # 3) forzar orden de columnas igual al esperado
    # Algunas versiones de scikit pueden dar np.ndarray; convertir a list
    expected_raw = list(expected_raw)
    # Solo mantener columnas que existan en df
    ordered = [c for c in expected_raw if c in df.columns]
    # añadir el resto que pueda haber (no romper)
    rest = [c for c in df.columns if c not in ordered]
    df = df[ordered + rest]

    # 4) tipos: forzar numerics a float
    for c in df.select_dtypes(include=["int64","int32"]).columns:
        df[c] = df[c].astype(float)

    return df

# Usar la función antes de predecir
df_forecast_aligned = align_forecast_with_preprocessor(df_forecast, model, region_name=region)

# Ahora predecir con el pipeline
df_forecast["pred_dem"] = model.predict(df_forecast_aligned)


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

# ---------------------------
# 🔍 Importancia de características (Top 10 Features)
# ---------------------------
st.subheader(f"Importancia de características para la región '{region}'")

try:
    estimator = model.named_steps.get("model", model)

    if hasattr(estimator, "feature_importances_"):
        importances = estimator.feature_importances_

        # Obtener los nombres reales del preprocesador dentro del pipeline
        feature_names = None
        if "preprocessing" in model.named_steps:
            preprocessor = model.named_steps["preprocessing"]
            if hasattr(preprocessor, "get_feature_names_out"):
                try:
                    feature_names = preprocessor.get_feature_names_out()
                    # Limpiar prefijos tipo "num__" o "cat__"
                    feature_names = [
                        name.split("__")[-1].replace("num_", "").replace("cat_", "")
                        for name in feature_names
                    ]
                except Exception:
                    feature_names = None

        # Si no los pudo obtener, usar nombres del modelo entrenado
        if feature_names is None:
            # usar columnas del modelo entrenado (más exacto)
            feature_names = getattr(estimator, "feature_names_in_", None)

        # Si todavía no hay nombres, usar columnas del DataFrame
        if feature_names is None:
            feature_names = list(df_forecast.columns)

        # Asegurar longitudes coherentes
        n_feats = len(importances)
        feature_names = list(feature_names)[:n_feats]
        if len(feature_names) < n_feats:
            feature_names += [f"Feature_{i}" for i in range(len(feature_names), n_feats)]

        # Crear dataframe ordenado por importancia
        fi = (
            pd.DataFrame({
                "feature": feature_names,
                "importance": importances
            })
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

        # Mostrar top 10
        st.write("Estas son las 10 variables más relevantes del modelo entrenado:")
        st.dataframe(fi.head(10))

        # Gráfico interactivo
        chart3 = (
            alt.Chart(fi.head(10))
            .mark_bar()
            .encode(
                x=alt.X("importance:Q", title="Importancia"),
                y=alt.Y("feature:N", sort="-x", title="Variable"),
                color=alt.Color("importance:Q", scale=alt.Scale(scheme="blues")),
                tooltip=["feature", "importance"]
            )
            .properties(height=350)
        )
        st.altair_chart(chart3, use_container_width=True)

    else:
        modelo_tipo = type(estimator).__name__
        st.info(f"El modelo para región '{region}' ({modelo_tipo}) no tiene atributo 'feature_importances_'.")
except Exception as e:
    st.warning(f"No se pudo mostrar importancias: {e}")


# ---------------------------
# DESCARGA
# ---------------------------
csv = df_forecast.to_csv(index=False)
st.download_button("Descargar predicciones (CSV)", csv, file_name=f"predicciones_{region}.csv", mime="text/csv")

st.success("Predicción completada correctamente.")
