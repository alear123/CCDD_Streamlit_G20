import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import altair as alt
import os
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta


st.set_page_config(layout="wide", page_title="Predicción de demanda eléctrica")

REGION_COORDS = {
    "edelap": {"lat": -34.921, "lon": -57.954},  
    "edesur": {"lat": -34.615, "lon": -58.425},  
    "edenor": {"lat": -34.567, "lon": -58.447}   
}

MODEL_FOLDER = "models"

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
    
from datetime import datetime, timedelta

def fetch_historical_demand(region_name, days_back):
    """
    Obtiene la demanda eléctrica histórica desde la API de CAMMESA
    para una región específica y una cantidad de días hacia atrás.
    
    Parámetros:
    -----------
    region_name : str
        Nombre de la región ("edelap", "edesur" o "edenor")
    days_back : int
        Cantidad de días hacia atrás a consultar
    
    Retorna:
    --------
    pd.DataFrame
        DataFrame con columnas ['fecha', 'dem']
    """
    # Mapa de regiones a sus IDs en CAMMESA
    REGION_IDS = {
        "edelap": 1943,
        "edenor": 1077,
        "edesur": 1078
    }

    if region_name not in REGION_IDS:
        raise ValueError(f"Región '{region_name}' no reconocida. Usa: {list(REGION_IDS.keys())}")

    region_id = REGION_IDS[region_name]
    base_url = "https://api.cammesa.com/demanda-svc/demanda/ObtieneDemandaYTemperaturaRegionByFecha"
    
    # Lista para acumular resultados diarios
    all_records = []

    for i in range(days_back):
        fecha_consulta = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        params = {"fecha": fecha_consulta, "id_region": region_id}
        try:
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                continue

            df_dia = pd.DataFrame(data)
            # Normalizamos los nombres y filtramos columnas importantes
            if "fecha" in df_dia.columns and "dem" in df_dia.columns:
                df_dia["fecha"] = pd.to_datetime(df_dia["fecha"], errors="coerce")
                all_records.append(df_dia[["fecha", "dem"]])
            else:
                # Algunos endpoints devuelven 'demanda' o similar
                posibles_cols = [c for c in df_dia.columns if "dem" in c.lower()]
                if posibles_cols:
                    df_dia["fecha"] = pd.to_datetime(df_dia["fecha"], errors="coerce")
                    df_dia = df_dia.rename(columns={posibles_cols[0]: "dem"})
                    all_records.append(df_dia[["fecha", "dem"]])
        except Exception as e:
            print(f"Error obteniendo datos del {fecha_consulta}: {e}")
            continue

    if not all_records:
        st.warning(f"No se obtuvieron datos históricos para {region_name}.")
        return pd.DataFrame(columns=["fecha", "dem"])

    # Concatenamos y ordenamos
    df_hist = pd.concat(all_records).dropna(subset=["fecha", "dem"]).sort_values("fecha")
    df_hist.reset_index(drop=True, inplace=True)
    return df_hist


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

st.markdown(
    """
    #  Predicción de Demanda Eléctrica por Región

    Bienvenido a la herramienta de predicción de demanda eléctrica.  
    Esta aplicación permite:
    
    - Obtener el pronóstico horario de demanda eléctrica para las principales regiones.
    - Visualizar la relación entre temperatura y demanda.
    - Consultar la importancia de las variables que influyen en la predicción.
    - Descargar los resultados para análisis posterior.
    """
)

# === Barra lateral ===
st.sidebar.title("Configuración")
region = st.sidebar.selectbox("Selecciona la región:", list(REGION_COORDS.keys()))
forecast_days = st.sidebar.slider("Días a predecir:", 1, 14, 7)

model = load_model(region)
if model is None:
    st.stop()

coords = REGION_COORDS[region]

# === Crear pestañas ===
tab_pred, tab_explore = st.tabs([" Predicción", " Análisis Exploratorio"])

# =====================================================
# === PESTAÑA 1: PREDICCIÓN ===========================
# =====================================================
with tab_pred:
    with st.spinner("Obteniendo pronóstico meteorológico..."):
        df_forecast = fetch_open_meteo_forecast(coords["lat"], coords["lon"], forecast_days=forecast_days)

    df_forecast_aligned = align_forecast(df_forecast, region)

    with st.spinner("Obteniendo datos históricos de CAMMESA..."):
        df_hist = fetch_historical_demand(region, days_back=forecast_days) 

    with st.spinner("Generando predicciones..."):
        df_forecast["pred_dem"] = model.predict(df_forecast_aligned)

    st.subheader(f"Resumen de la demanda para '{region}'")
    col1, col2, col3 = st.columns(3)
    col1.metric("Máx. demanda", f"{df_forecast['pred_dem'].max():.2f} MW")
    col2.metric("Mín. demanda", f"{df_forecast['pred_dem'].min():.2f} MW")
    col3.metric("Promedio", f"{df_forecast['pred_dem'].mean():.2f} MW")

    # --- Demanda histórica y predicción combinadas ---
    df_hist["fecha"] = pd.to_datetime(df_hist["fecha"], errors="coerce")
    df_hist = (
        df_hist.set_index("fecha")
        .resample("1H")
        .mean(numeric_only=True)
        .dropna(subset=["dem"])
        .reset_index()
    )

    if not df_hist.empty:
        df_hist["tipo"] = "Histórico"
        df_forecast_rename = df_forecast.rename(columns={"pred_dem": "dem"}).copy()
        df_forecast_rename["tipo"] = "Predicción"

        last_hist_date = df_hist["fecha"].max()
        forecast_start = last_hist_date + timedelta(hours=1)
        df_forecast_rename = df_forecast_rename.sort_values("fecha").reset_index(drop=True)
        df_forecast_rename["fecha"] = [
            forecast_start + timedelta(hours=i)
            for i in range(len(df_forecast_rename))
        ]

        df_comb = pd.concat(
            [df_hist[["fecha", "dem", "tipo"]], df_forecast_rename[["fecha", "dem", "tipo"]]],
            ignore_index=True
        ).dropna(subset=["fecha", "dem"])

        base = alt.Chart(df_comb).encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("dem:Q", title="Demanda (MW)"),
            color=alt.Color("tipo:N", title="Tipo de datos",
                            scale=alt.Scale(domain=["Histórico", "Predicción"],
                                            range=["gray", "blue"])),
            tooltip=["fecha:T", "dem:Q", "tipo:N"]
        )

        chart_comb = base.mark_line(point=False, strokeWidth=2).interactive().properties(
            title="Demanda histórica y predicción combinadas"
        )

        st.altair_chart(chart_comb, use_container_width=True)
    else:
        st.info("No se encontraron datos históricos para la región seleccionada.")

    # --- Temperatura vs Demanda ---
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

    # --- Distribución horaria ---
    st.subheader(f"Distribución horaria de demanda para '{region}'")
    df_forecast['hora'] = df_forecast['fecha'].dt.hour
    chart_box = alt.Chart(df_forecast).mark_boxplot(extent='min-max').encode(
        x=alt.X("hora:O", title="Hora del día"),
        y=alt.Y("pred_dem:Q", title="Demanda (MW)")
    ).properties(height=400)
    st.altair_chart(chart_box, use_container_width=True)

    # --- Descarga ---
    csv = df_forecast.to_csv(index=False)
    st.download_button(" Descargar predicciones (CSV)", csv,
                       file_name=f"predicciones_{region}.csv", mime="text/csv")
    st.success("Predicción completada correctamente.")

# =====================================================
# === PESTAÑA 2: ANÁLISIS EXPLORATORIO ===============
# =====================================================
with tab_explore:
    import altair as alt
    alt.data_transformers.disable_max_rows()

    st.header("📊 Análisis Exploratorio de Datos (EDA)")
    st.info("Explorá las relaciones entre variables climáticas y la demanda energética utilizando visualizaciones interactivas.")

    # 📂 Cargar dataset local
    df = pd.read_csv("dataset/master_energy_preprocessed.csv")

    # Conversión de fechas y columna hora
    df["fecha"] = pd.to_datetime(df["fecha"])
    if "hora" not in df.columns:
        df["hora"] = df["fecha"].dt.hour

    st.write(f"**Filas:** {df.shape[0]} | **Columnas:** {df.shape[1]}")
    st.dataframe(df.head())

    # ========================================================
    # 🔹 FILTROS INTERACTIVOS
    # ========================================================
    st.sidebar.subheader("Filtros")
    regiones = sorted(df["region"].unique().tolist())
    estaciones = sorted(df["estacion"].unique().tolist())

    region_sel = st.sidebar.selectbox("🌍 Región", opciones := regiones)
    estacion_sel = st.sidebar.multiselect("🌦 Estación", estaciones, default=estaciones)
    fecha_min, fecha_max = df["fecha"].min(), df["fecha"].max()
    rango_fechas = st.sidebar.slider(
        "🗓 Rango de fechas",
        min_value=fecha_min.to_pydatetime(),
        max_value=fecha_max.to_pydatetime(),
        value=(fecha_min.to_pydatetime(), fecha_max.to_pydatetime()),
        format="YYYY-MM-DD"
    )

    # Aplicar filtros
    df_filtrado = df[
        (df["region"] == region_sel)
        & (df["estacion"].isin(estacion_sel))
        & (df["fecha"].between(rango_fechas[0], rango_fechas[1]))
    ]

    st.write(f"📍 Mostrando datos para **{region_sel}**, estaciones {', '.join(estacion_sel)} "
             f"entre {rango_fechas[0].strftime('%Y-%m-%d')} y {rango_fechas[1].strftime('%Y-%m-%d')}")

    # ========================================================
    # 🔹 GRÁFICO 1: Dispersión seleccionable (exploración libre)
    # ========================================================
    st.subheader("🔍 Exploración de relaciones entre variables")

    columnas_numericas = df.select_dtypes(include=["float64", "int64"]).columns.tolist()
    x_var = st.selectbox("Variable en eje X", columnas_numericas, index=columnas_numericas.index("temperature_2m"))
    y_var = st.selectbox("Variable en eje Y", columnas_numericas, index=columnas_numericas.index("dem"))

    chart_disp = (
        alt.Chart(df_filtrado)
        .mark_circle(size=60, opacity=0.6)
        .encode(
            x=alt.X(f"{x_var}:Q", title=x_var),
            y=alt.Y(f"{y_var}:Q", title=y_var),
            color=alt.Color("estacion:N", title="Estación"),
            tooltip=["fecha:T", "region:N", "estacion:N", f"{x_var}:Q", f"{y_var}:Q"]
        )
        .properties(
            title=f"Relación entre {x_var} y {y_var}",
            width=700,
            height=400
        )
        .interactive()
    )
    st.altair_chart(chart_disp, use_container_width=True)

    # ========================================================
    # 🔹 GRÁFICO 2: Evolución temporal de la demanda
    # ========================================================
    st.subheader("⏱ Evolución temporal de la demanda energética")

    chart_linea = (
        alt.Chart(df_filtrado)
        .mark_line()
        .encode(
            x=alt.X("fecha:T", title="Fecha"),
            y=alt.Y("dem:Q", title="Demanda (MW)"),
            color=alt.Color("estacion:N", title="Estación"),
            tooltip=["fecha:T", "dem:Q", "estacion:N"]
        )
        .properties(
            width=700,
            height=400,
            title="Demanda energética a lo largo del tiempo"
        )
        .interactive()
    )
    st.altair_chart(chart_linea, use_container_width=True)

    # ========================================================
    # 🔹 GRÁFICO 3: Matriz de correlación
    # ========================================================
    st.subheader("📈 Correlación entre variables numéricas")

    corr = df_filtrado[columnas_numericas].corr().reset_index().melt("index")
    corr_chart = (
        alt.Chart(corr)
        .mark_rect()
        .encode(
            x=alt.X("index:N", title="Variable X"),
            y=alt.Y("variable:N", title="Variable Y"),
            color=alt.Color("value:Q", scale=alt.Scale(scheme="blueorange", domain=(-1, 1)), title="Correlación"),
            tooltip=["index:N", "variable:N", alt.Tooltip("value:Q", format=".2f")]
        )
        .properties(width=600, height=600, title="Matriz de correlación")
    )
    st.altair_chart(corr_chart, use_container_width=True)
