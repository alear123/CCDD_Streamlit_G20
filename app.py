# ==================== CONFIGURACIÓN UI ====================
st.set_page_config(
    layout="wide",
    page_title="Predicción de Demanda Eléctrica ⚡",
    page_icon="⚡"
)

st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        padding: 20px;
        border-radius: 12px;
    }
    h1, h2, h3 {
        color: #1E3A8A;
        font-weight: 700;
    }
    .stMetric {
        background-color: white !important;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }
    hr {
        border: none;
        border-top: 2px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown(
    """
    # ⚡ Predicción de Demanda Eléctrica por Región

    <p style="font-size: 18px; color: #374151;">
    Esta herramienta te permite analizar y predecir la demanda energética de forma interactiva.  
    </p>

    **Funciones principales:**
    - 🔮 Predicción horaria de demanda eléctrica.
    - 🌦 Análisis del impacto del clima (temperatura, viento, humedad).
    - 📉 Visualización histórica y comparación con pronóstico.
    - 📁 Descarga de resultados para análisis posterior.
    """,
    unsafe_allow_html=True
)

# === SIDEBAR ===
with st.sidebar:
    st.markdown("## ⚙️ Configuración")
    region = st.selectbox("🌍 Selecciona la región:", list(REGION_COORDS.keys()))
    forecast_days = st.slider("🗓️ Días a predecir:", 1, 14, 7)
    st.markdown("---")
    st.info("💡 Consejo: Cuantos más días selecciones, mayor será el rango de predicción.")

# === PESTAÑAS ===
tab_pred, tab_explore = st.tabs(["🔮 Predicción", "📊 Análisis Exploratorio"])

# =====================================================
# === TAB 1: PREDICCIÓN ===============================
# =====================================================
with tab_pred:
    st.markdown("## 🔮 Predicción de demanda energética")

    with st.spinner("📡 Obteniendo pronóstico meteorológico..."):
        df_forecast = fetch_open_meteo_forecast(coords["lat"], coords["lon"], forecast_days=forecast_days)
    df_forecast_aligned = align_forecast(df_forecast, region)

    with st.spinner("⚙️ Cargando modelo..."):
        model = load_model(region)
        if model is None:
            st.stop()

    with st.spinner("📈 Generando predicciones..."):
        df_forecast["pred_dem"] = model.predict(df_forecast_aligned)

    # --- Métricas visuales ---
    st.markdown("### 📊 Resumen de la demanda")
    col1, col2, col3 = st.columns(3)
    col1.metric("🔝 Máxima demanda", f"{df_forecast['pred_dem'].max():.2f} MW")
    col2.metric("🔻 Mínima demanda", f"{df_forecast['pred_dem'].min():.2f} MW")
    col3.metric("⚖️ Promedio", f"{df_forecast['pred_dem'].mean():.2f} MW")

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Gráfico histórico + predicción ---
    st.subheader("📈 Demanda histórica vs predicción")
    st.caption("Comparación entre los valores históricos de CAMMESA y la predicción generada por el modelo.")

    # (Mantener aquí tu código para generar df_comb y graficar con Altair)
    # ...
    st.altair_chart(chart_comb, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Temperatura vs Demanda ---
    st.subheader("🌡️ Relación temperatura - demanda")
    st.caption("Se observa cómo la temperatura influye directamente en la demanda energética.")
    st.altair_chart(chart2, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Distribución horaria ---
    st.subheader("⏰ Distribución horaria de la demanda")
    st.caption("Análisis de la variación de la demanda según la hora del día.")
    st.altair_chart(chart_box, use_container_width=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # --- Descarga ---
    csv = df_forecast.to_csv(index=False)
    st.download_button(
        "💾 Descargar predicciones (CSV)",
        csv,
        file_name=f"predicciones_{region}.csv",
        mime="text/csv",
        use_container_width=True
    )
    st.success("✅ Predicción completada correctamente.")

# =====================================================
# === TAB 2: ANÁLISIS EXPLORATORIO ====================
# =====================================================
with tab_explore:
    st.markdown("## 📊 Análisis Exploratorio de Datos (EDA)")
    st.info("Explorá las relaciones entre las variables climáticas y la demanda energética.")

    # Mantener tus gráficos (Temperatura vs Demanda, Patrón Horario, Viento, etc.)
    # Recomendación visual: cambia los títulos por emojis + nombres más cortos
    st.subheader("🌡️ Temperatura vs Demanda")
    st.subheader("⏰ Patrón horario de la demanda")
    st.subheader("🌬️ Viento vs Demanda")
