import streamlit as st
import pandas as pd
import numpy as np
import requests
import joblib
import altair as alt
import os
from sklearn.base import BaseEstimator, TransformerMixin
from datetime import datetime, timedelta

# Configuración de página con tema personalizado
st.set_page_config(
    layout="wide", 
    page_title="Predicción de Demanda Eléctrica",
    page_icon="",
    initial_sidebar_state="expanded"
)

# CSS personalizado para mejorar la UI
st.markdown("""
    <style>
    /* Mejorar el encabezado principal */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.5rem;
        font-weight: 700;
    }
    
    .main-header p {
        margin: 0.5rem 0 0 0;
        font-size: 1.1rem;
        opacity: 0.95;
    }
    
    /* Mejorar las métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Mejorar las tarjetas */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8f9fa 0%, #ffffff 100%);
    }
    
    /* Botones */
    .stDownloadButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 10px;
        border-left: 4px solid #667eea;
    }
    
    /* Secciones */
    .section-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

REGION_COORDS = {
    "edelap": {"lat": -34.921, "lon": -57.954, "nombre": "EDELAP - La Plata"},
    "edesur": {"lat": -34.615, "lon": -58.425, "nombre": "EDESUR - Buenos Aires Sur"},
    "edenor": {"lat": -34.567, "lon": -58.447, "nombre": "EDENOR - Buenos Aires Norte"}
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

def fetch_historical_demand(region_name, days_back):
    REGION_IDS = {
        "edelap": 1943,
        "edenor": 1077,
        "edesur": 1078
    }

    if region_name not in REGION_IDS:
        raise ValueError(f"Región '{region_name}' no reconocida. Usa: {list(REGION_IDS.keys())}")

    region_id = REGION_IDS[region_name]
    base_url = "https://api.cammesa.com/demanda-svc/demanda/ObtieneDemandaYTemperaturaRegionByFecha"
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
            if "fecha" in df_dia.columns and "dem" in df_dia.columns:
                df_dia["fecha"] = pd.to_datetime(df_dia["fecha"], errors="coerce")
                all_records.append(df_dia[["fecha", "dem"]])
            else:
                posibles_cols = [c for c in df_dia.columns if "dem" in c.lower()]
                if posibles_cols:
                    df_dia["fecha"] = pd.to_datetime(df_dia["fecha"], errors="coerce")
                    df_dia = df_dia.rename(columns={posibles_cols[0]: "dem"})
                    all_records.append(df_dia[["fecha", "dem"]])
        except Exception as e:
            continue

    if not all_records:
        st.warning(f"⚠️ No se obtuvieron datos históricos para {region_name}.")
        return pd.DataFrame(columns=["fecha", "dem"])

    df_hist = pd.concat(all_records).dropna(subset=["fecha", "dem"]).sort_values("fecha")
    df_hist.reset_index(drop=True, inplace=True)
    return df_hist

def load_model(region, model_folder=MODEL_FOLDER):
    model_path = os.path.join(model_folder, f"model_{region}.pkl")
    if not os.path.exists(model_path):
        st.error(f" No se encontró el modelo para la región '{region}' en {model_path}")
        return None
    try:
        return joblib.load(model_path)
    except Exception as e:
        st.error(f" Error al cargar el modelo: {e}")
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

# Header principal mejorado
st.markdown("""
    <div class="main-header">
        <h1> Predicción de Demanda Eléctrica</h1>
        <p>Sistema inteligente de pronóstico para regiones de Buenos Aires y La Plata</p>
    </div>
""", unsafe_allow_html=True)

# Sidebar mejorado
with st.sidebar:
    st.title(" Configuración")
    st.markdown("---")
    
    region = st.selectbox(
        " Región",
        list(REGION_COORDS.keys()),
        format_func=lambda x: REGION_COORDS[x]["nombre"]
    )
    
    forecast_days = st.slider(
        " Días a predecir",
        min_value=1,
        max_value=14,
        value=7,
        help="Selecciona cuántos días deseas pronosticar"
    )
    
    st.markdown("---")
    st.markdown("###  Información")
    st.info(f"""
    **Región:** {REGION_COORDS[region]["nombre"]}  
    **Coordenadas:** {REGION_COORDS[region]["lat"]}, {REGION_COORDS[region]["lon"]}  
    **Días:** {forecast_days}
    """)

model = load_model(region)
if model is None:
    st.stop()

coords = REGION_COORDS[region]

# Tabs mejoradas
tab_pred, tab_explore = st.tabs([" Predicción y Análisis", " Análisis Exploratorio"])

# =====================================================
# PESTAÑA 1: PREDICCIÓN
# =====================================================
with tab_pred:
    with st.spinner(" Obteniendo pronóstico meteorológico..."):
        df_forecast = fetch_open_meteo_forecast(coords["lat"], coords["lon"], forecast_days=forecast_days)

    df_forecast_aligned = align_forecast(df_forecast, region)

    with st.spinner(" Obteniendo datos históricos de CAMMESA..."):
        df_hist = fetch_historical_demand(region, days_back=forecast_days) 

    with st.spinner(" Generando predicciones"):
        df_forecast["pred_dem"] = model.predict(df_forecast_aligned)

    # Métricas destacadas
    st.markdown("###  Resumen ")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            " Demanda Máxima",
            f"{df_forecast['pred_dem'].max():.0f} MW",
            delta=f"{((df_forecast['pred_dem'].max() - df_forecast['pred_dem'].mean()) / df_forecast['pred_dem'].mean() * 100):.1f}%"
        )
    
    with col2:
        st.metric(
            " Demanda Mínima",
            f"{df_forecast['pred_dem'].min():.0f} MW",
            delta=f"-{((df_forecast['pred_dem'].mean() - df_forecast['pred_dem'].min()) / df_forecast['pred_dem'].mean() * 100):.1f}%"
        )
    
    with col3:
        st.metric(
            " Promedio",
            f"{df_forecast['pred_dem'].mean():.0f} MW"
        )
    
    with col4:
        st.metric(
            " Temp. Promedio",
            f"{df_forecast['temperature_2m'].mean():.1f}°C"
        )

    st.markdown("---")

    # Gráfico principal combinado
    st.markdown("###  Demanda Histórica y Predicción")
    
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
            x=alt.X("fecha:T", title="Fecha y Hora", axis=alt.Axis(format="%d/%m %H:%M")),
            y=alt.Y("dem:Q", title="Demanda (MW)", scale=alt.Scale(zero=False)),
            color=alt.Color("tipo:N", 
                          title="Serie",
                          scale=alt.Scale(domain=["Histórico", "Predicción"],
                                        range=["#667eea", "#f093fb"])),
            tooltip=[
                alt.Tooltip("fecha:T", title="Fecha", format="%d/%m/%Y %H:%M"),
                alt.Tooltip("dem:Q", title="Demanda (MW)", format=".2f"),
                alt.Tooltip("tipo:N", title="Tipo")
            ]
        )

        chart_comb = base.mark_line(strokeWidth=3).interactive().properties(
            height=450,
            title={
                "text": "Evolución de la Demanda Eléctrica",
                "fontSize": 18,
                "fontWeight": "bold"
            }
        )

        st.altair_chart(chart_comb, use_container_width=True)
    else:
        st.info(" No se encontraron datos históricos para la región seleccionada.")

    # Dos columnas para gráficos adicionales
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("###  Temperatura vs Demanda")
        
        df_forecast['hora'] = df_forecast['fecha'].dt.hour
        
        base_temp = alt.Chart(df_forecast).encode(
            x=alt.X("fecha:T", title="Fecha", axis=alt.Axis(format="%d/%m")),
        )
        
        line_temp = base_temp.mark_line(color="#ff6b6b", strokeWidth=2).encode(
            y=alt.Y("temperature_2m:Q", title="Temperatura (°C)"),
            tooltip=[
                alt.Tooltip("fecha:T", format="%d/%m %H:%M"),
                alt.Tooltip("temperature_2m:Q", title="Temp (°C)", format=".1f")
            ]
        )
        
        line_dem = base_temp.mark_line(color="#4ecdc4", strokeWidth=2).encode(
            y=alt.Y("pred_dem:Q", title="Demanda (MW)"),
            tooltip=[
                alt.Tooltip("fecha:T", format="%d/%m %H:%M"),
                alt.Tooltip("pred_dem:Q", title="Demanda (MW)", format=".2f")
            ]
        )
        
        chart_combined = alt.layer(line_temp, line_dem).resolve_scale(
            y="independent"
        ).properties(height=350).interactive()
        
        st.altair_chart(chart_combined, use_container_width=True)
    
    with col_right:
        st.markdown("### Distribución Horaria")
        
        chart_box = alt.Chart(df_forecast).mark_boxplot(
            extent='min-max',
            color="#667eea",
            opacity=0.7
        ).encode(
            x=alt.X("hora:O", title="Hora del día"),
            y=alt.Y("pred_dem:Q", title="Demanda (MW)"),
            tooltip=["hora:O", "pred_dem:Q"]
        ).properties(height=350)
        
        st.altair_chart(chart_box, use_container_width=True)

    # Botón de descarga mejorado
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        csv = df_forecast.to_csv(index=False)
        st.download_button(
            " Descargar Predicciones Completas",
            csv,
            file_name=f"predicciones_{region}_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    st.success(" Predicción completada correctamente")

# =====================================================
# PESTAÑA 2: ANÁLISIS EXPLORATORIO
# =====================================================
with tab_explore:
    alt.data_transformers.disable_max_rows()

    st.markdown("##  Análisis Exploratorio de Datos")
    st.info(" Explora las relaciones entre variables climáticas y la demanda energética con visualizaciones interactivas.")

    df = pd.read_csv("dataset/master_energy_preprocessed.csv")
    df["fecha"] = pd.to_datetime(df["fecha"])
    if "hora" not in df.columns:
        df["hora"] = df["fecha"].dt.hour

    # Gráfico 1: Temperatura vs Demanda
    st.markdown("###  Temperatura vs Demanda Energética")
    
    region_param = alt.param(
        name='Región',
        bind=alt.binding_select(
            options=list(df['region'].unique()),
            name='Región: '
        ),
        value=df['region'].unique()[0]
    )

    estacion_param = alt.param(
        name='Estación',
        bind=alt.binding_select(
            options=['Todas'] + sorted(df['estacion'].unique().tolist()),
            name='Estación: '
        ),
        value='Todas'
    )

    chart_temp_dem = (
        alt.Chart(df)
        .mark_circle(size=80, opacity=0.5)
        .encode(
            x=alt.X('temperature_2m:Q', title='Temperatura (°C)'),
            y=alt.Y('dem:Q', title='Demanda energética (MW)'),
            color=alt.Color('estacion:N', 
                          title='Estación',
                          scale=alt.Scale(scheme='category10')),
            tooltip=[
                alt.Tooltip('fecha:T', format='%d/%m/%Y %H:%M'),
                alt.Tooltip('temperature_2m:Q', title='Temp (°C)', format='.1f'),
                alt.Tooltip('dem:Q', title='Demanda (MW)', format='.2f'),
                'region:N',
                'estacion:N'
            ]
        )
        .add_params(region_param, estacion_param)
        .transform_filter('datum.region == Región')
        .transform_filter("(Estación == 'Todas') || (datum.estacion == Estación)")
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart_temp_dem, use_container_width=True)

    st.markdown("---")

    # Gráfico 2: Patrón horario
    st.markdown("###  Patrón Horario de la Demanda")

    region_param_hora = alt.param(
        name='RegiónHora',
        bind=alt.binding_select(
            options=list(df['region'].unique()),
            name='Región: '
        ),
        value=df['region'].unique()[0]
    )

    df_horario = (
        df.groupby(["region", "hora", "fin_de_semana"])["dem"]
        .mean()
        .reset_index()
    )

    df_horario["tipo_dia"] = df_horario["fin_de_semana"].map({0: "Día laboral", 1: "Fin de semana"})

    chart_horario = (
        alt.Chart(df_horario)
        .mark_line(point=True, strokeWidth=3)
        .encode(
            x=alt.X("hora:O", title="Hora del día"),
            y=alt.Y("dem:Q", title="Demanda promedio (MW)"),
            color=alt.Color("tipo_dia:N", 
                          title="Tipo de día",
                          scale=alt.Scale(domain=["Día laboral", "Fin de semana"],
                                        range=["#667eea", "#f093fb"])),
            tooltip=["hora:O", 
                    alt.Tooltip("dem:Q", format=".2f", title="Demanda (MW)"),
                    "tipo_dia:N"]
        )
        .add_params(region_param_hora)
        .transform_filter("datum.region == RegiónHora")
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart_horario, use_container_width=True)

    st.markdown("---")

    # Gráfico 3: Viento vs Demanda
    st.markdown("###  Velocidad del Viento vs Demanda")

    region_param_viento = alt.param(
        name='RegiónViento',
        bind=alt.binding_select(
            options=list(df['region'].unique()),
            name='Región: '
        ),
        value=df['region'].unique()[0]
    )

    estacion_param_viento = alt.param(
        name='EstacionViento',
        bind=alt.binding_select(
            options=['Todas'] + sorted(df['estacion'].unique().tolist()),
            name='Estación: '
        ),
        value='Todas'
    )

    chart_viento = (
        alt.Chart(df)
        .mark_circle(size=80, opacity=0.5)
        .encode(
            x=alt.X('wind_speed_10m:Q', title='Velocidad del viento (m/s)'),
            y=alt.Y('dem:Q', title='Demanda energética (MW)'),
            color=alt.Color('estacion:N', 
                          title='Estación',
                          scale=alt.Scale(scheme='category10')),
            tooltip=[
                alt.Tooltip('fecha:T', format='%d/%m/%Y %H:%M'),
                alt.Tooltip('wind_speed_10m:Q', title='Viento (m/s)', format='.1f'),
                alt.Tooltip('dem:Q', title='Demanda (MW)', format='.2f'),
                'region:N',
                'estacion:N'
            ]
        )
        .add_params(region_param_viento, estacion_param_viento)
        .transform_filter("datum.region == RegiónViento")
        .transform_filter("(EstacionViento == 'Todas') || (datum.estacion == EstacionViento)")
        .properties(height=450)
        .interactive()
    )

    st.altair_chart(chart_viento, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p> Sistema de Predicción de Demanda Eléctrica </p>
        <p style='font-size: 0.9rem;'>Datos provistos por CAMMESA y Open-Meteo</p>
    </div>
""", unsafe_allow_html=True)