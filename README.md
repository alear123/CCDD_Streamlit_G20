<img width="1838" height="735" alt="image" src="https://github.com/user-attachments/assets/9a1bb353-61d7-4566-ab6d-82080ee596fe" />Energy Demand Predictor: Buenos Aires ⚡

Este proyecto utiliza **Machine Learning** para predecir la demanda energética horaria en las regiones de **Edenor, Edesur y Edelap**. A través de la integración de datos históricos de **CAMMESA** y variables climáticas de **Open-Meteo**, el modelo permite optimizar la generación, reducir costos y prevenir sobrecargas en la red eléctrica.
<img width="1919" height="882" alt="image" src="https://github.com/user-attachments/assets/1b2b6d71-8534-4575-8c84-2e71e04ff2ef" />
<img width="1846" height="555" alt="image" src="https://github.com/user-attachments/assets/0a015bd3-89f5-470e-8602-6b62c016cf32" />
<img width="1838" height="735" alt="image" src="https://github.com/user-attachments/assets/e4ff8b9b-f806-4e2f-b8c1-95e5e15c41a5" />




---

## 🛠️ Stack Tecnológico

* **Data Orchestration:** Apache Airflow (Astro CLI).
* **Machine Learning:** Scikit-Learn (Ridge, Random Forest, Gradient Boosting).
* **Data Analysis:** Pandas, NumPy, Matplotlib, Seaborn.
* **Notebooks:** Google Colab.
* **Interface:** Streamlit.
* **APIs:** Open-Meteo & CAMMESA Scraper.

---

## 🏗️ Arquitectura del Proyecto

El flujo de trabajo se divide en cuatro etapas críticas:

### 1. Ingesta y Orquestación (ETL)
Utilizamos **Apache Airflow** mediante **Astro CLI** para automatizar la recolección de datos de los últimos 247 días:
* **Demanda:** Datos históricos de CAMMESA de 3 regiones de Buenos Aires.
* **Clima:** Consumo de la API de Open-Meteo (temperatura a 2m, presión al nivel del mar, velocidad del viento).
* **Matching:** Cruce de datos por estampa de tiempo para generar un dataset unificado y consistente.

### 2. Análisis Exploratorio de Datos (EDA)
Realizado en **Google Colab**, el objetivo fue identificar la correlación entre variables climáticas y picos de demanda.
* **Inspección:** Revisión de estructura, tipos y nulos en variables clave.
* **Limpieza:** Interpolación y relleno de valores faltantes.
* **Feature Engineering:** Creación de variables derivadas como `fin_de_semana` y `estacion` a partir de la fecha para capturar patrones estacionales y sociales.

### 3. Modelado y Entrenamiento
Planteamos el problema como una **Regresión** para estimar valores continuos de demanda energética.
* **Preprocesamiento:** Implementación de pipelines para escalado, imputación y codificación cíclica (para capturar la naturaleza horaria y mensual).
* **Modelos Evaluados:**
    * **Lineales:** Ridge, Lasso, ElasticNet, Regresión Lineal.
    * **No lineales:** Random Forest, Gradient Boosting.
* **Optimización:** Búsqueda de hiperparámetros mediante **GridSearchCV**.

#### Resultados Obtenidos (R²)
| Región | Precisión (R²) |
| :--- | :--- |
| **Edelap** | 0.93 |
| **Edesur** | 0.94 |
| **Edenor** | 0.93 |

### 4. Visualización
Despliegue de una interfaz interactiva en **Streamlit** donde se pueden visualizar las predicciones y el comportamiento de las variables en tiempo real.

---

## 🚀 Instalación y Uso

1. **Clonar el repositorio:**
   git clone [https://github.com/alear123/CCDD_Streamlit_G20](https://github.com/alear123/CCDD_Streamlit_G20)

2. Levantar el orquestador (Airflow):
    astro dev start

3. Ejecutar la interfaz de Streamlit:
    pip install -r requirements.txt
    streamlit run app.py

O hacer click acá: https://ccddappg20-cypow9vvrx6zvcadssbjmk.streamlit.app/

👥 Colaboradores
Este es un proyecto grupal desarrollado por:

Alejandro Arce

Enzo Silva
