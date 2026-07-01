# JV Program — Sistema de Control de Producción VLR México

**Trabajo Fin de Máster — Máster Universitario en Análisis y Visualización de Datos Masivos (UNIR)**
**Equipo 1_F** · 2026

Repositorio del proyecto de análisis de datos aplicado al sistema de control de producción JV Program de VLR México, empresa maquiladora textil especializada en la confección de muebles tapizados para exportación.

---

## 📁 Estructura del repositorio

```
JV_Program/
├── etl/
│   └── 01_ETL_JVProgram.py        # Extracción, transformación y carga de datos
├── eda/
│   └── 02_EDA_Complementario.py   # Análisis exploratorio complementario
├── modeling/
│   ├── 02_Modelado_JVProgram.py   # Entrenamiento y evaluación del modelo
│   └── modelos/                   # Modelos serializados (.pkl) — generados al ejecutar
├── dashboard/
│   ├── dashboard_app.py           # Aplicación Streamlit
│   └── requirements.txt           # Dependencias específicas del dashboard
├── figuras/
│   ├── 03_Figuras_Capitulo6_7.py  # Figuras de Evaluación, Discusión y Conclusiones
│   └── 04_Figuras_Capitulo8.py    # Figuras del Prototipo (arquitectura, dashboard, modelo de datos)
├── data/
│   ├── raw/                       # (no versionado) datos crudos extraídos de SQL Server
│   └── processed/                 # (no versionado) datasets .parquet / .csv generados por el ETL
├── requirements.txt               # Dependencias generales del proyecto
└── README.md                      # Este archivo
```

---

## ⚙️ Instalación

### 1. Clonar el repositorio

```bash
git clone <URL_DEL_REPOSITORIO>
cd JV_Program
```

### 2. Crear entorno de conda (recomendado)

El proyecto requiere un entorno aislado para evitar conflictos de versiones entre NumPy, pandas y las librerías de machine learning.

```bash
conda create -n jvprogram python=3.11 -y
conda activate jvprogram

conda install numpy=1.26 pandas matplotlib seaborn scikit-learn scipy -y
pip install -r requirements.txt
```

### 3. Registrar el entorno en Jupyter (opcional, para correr los notebooks)

```bash
conda install jupyter ipykernel -y
python -m ipykernel install --user --name jvprogram --display-name "Python JVProgram"
```

---

## 🔌 Configuración de la conexión a SQL Server

El sistema JV Program almacena sus datos en una base de datos **DB_MASTER_PLAN** sobre Microsoft SQL Server Express, dentro de la red local de VLR México.

1. Abre `etl/01_ETL_JVProgram.py`
2. Ajusta la cadena de conexión en la Celda 2:

```python
SERVER   = r"NOMBRE_PC\SQLEXPRESS"   # ej. r"LAPTOP-VLR\SQLEXPRESS"
DATABASE = "DB_MASTER_PLAN"
```

3. Verifica que tienes instalado el **ODBC Driver 17 for SQL Server** (o superior) en el equipo donde se ejecuta el script.

---

## ▶️ Ejecución del prototipo — flujo completo

El prototipo se ejecuta en cuatro pasos secuenciales, replicando las capas de la arquitectura descrita en la Sección 8.1 del documento del TFM.

### Paso 1 — ETL (extracción y preparación de datos)

```bash
cd etl
jupyter notebook 01_ETL_JVProgram.py
```

Ejecutar todas las celdas en orden. Al finalizar, se generan en `data/processed/`:

- `dataset_WO_nivel.parquet` — dataset a nivel de orden de trabajo
- `dataset_QC_eventos.parquet` — dataset a nivel de evento de inspección de calidad

### Paso 2 — Modelado predictivo

```bash
cd ../modeling
jupyter notebook 02_Modelado_JVProgram.py
```

Requiere `xgboost`, `shap` y `joblib`:

```bash
pip install xgboost shap joblib
```

Al finalizar, se generan en `modeling/modelos/`:

- `modelo_xgb.pkl` — clasificador XGBoost optimizado
- `modelo_rf.pkl` — clasificador Random Forest (comparativo)
- `modelo_rl.pkl` — clasificador de Regresión Logística (baseline)
- `preprocessor.pkl` — pipeline de transformación de variables

### Paso 3 — Generación de figuras para el documento

```bash
cd ../figuras
jupyter notebook 03_Figuras_Capitulo6_7.py   # Figuras 23–27 (Cap. 6 y 7)
jupyter notebook 04_Figuras_Capitulo8.py     # Figuras 28–30 (Cap. 8, Prototipo)
```

### Paso 4 — Levantar el dashboard interactivo

```bash
cd ../dashboard
pip install -r requirements.txt
streamlit run dashboard_app.py
```

El dashboard se abrirá automáticamente en `http://localhost:8501`, con cuatro pestañas:

| Pestaña | Contenido |
|---|---|
| 📈 KPIs | Indicadores en tiempo real: volumen de WO, tasa de rechazo, eficiencia |
| 📉 Tendencias | Series temporales interactivas de producción y calidad |
| 🗺️ Mapa de calidad | Mapa de calor de defectos y diagrama de Pareto |
| 🎯 Predicción de riesgo | Formulario de predicción con el modelo XGBoost + explicación SHAP |

---

## 🧮 Modelo de datos

El prototipo se apoya en dos niveles de modelo de datos:

1. **Modelo relacional fuente** — esquema real de `DB_MASTER_PLAN` (SQL Server), documentado en la Sección 2.2 del TFM: `tbMaster_plan`, `tbCUT`, `tbKANBAN_PREP`, `tbSEWING`, `tbQC`, `tbSHIPPED` y tablas de catálogos.

2. **Modelo analítico (esquema en estrella)** — construido sobre los datasets Parquet para optimizar las consultas del dashboard: tabla de hechos `FACT_WO` y dimensiones `DIM_TIEMPO`, `DIM_PRODUCTO`, `DIM_OPERARIO`, `DIM_DEFECTO` (ver Figura 30 del documento).

---

## 📊 Resultados principales del modelo

| Métrica | Valor (conjunto de prueba holdout) |
|---|---|
| AUC-ROC | 0.9906 |
| Recall (umbral óptimo 0.45) | 0.9983 |
| Precision | 0.9949 |
| F1-Score | 0.9961 |
| Variable más importante | `BODY_PILLOW_REPAIR/SAMPLE` (70.6%) |

---

## ⚠️ Limitaciones del prototipo

- Ejecución en entorno local (`localhost`), sin autenticación de usuarios.
- Actualización de datos no automática — requiere re-ejecutar el ETL manualmente.
- Sin mecanismo de re-entrenamiento automatizado (recomendado: trimestral, ver Sección 6.6.2 del TFM).

---

## 👥 Equipo 1_F

- Adrian Andrade Guerrero
- Josué Enrique Juárez Coronado
- Francis Elieth Macias Barajas
- Fernando Rodelo Barron
- Jorge de Jesús Ventura Zapata

**Profesor:** Abel Alejandro Coronado Iruegas
**Asignatura:** Trabajo Fin de Máster — Máster Universitario en Análisis y Visualización de Datos Masivos
**Universidad Internacional de La Rioja (UNIR)** · 2026

---

## 📄 Licencia y confidencialidad

Los datos de VLR México utilizados en este proyecto son propiedad de la empresa y contienen información comercialmente sensible. Los datos de clientes han sido anonimizados (`CLI_001`, `CLI_002`...) y los identificadores de operario han sido sustituidos por códigos (`OP_001`, `OP_002`...) en todas las salidas públicas del proyecto. Este repositorio se comparte exclusivamente con fines académicos en el marco del Trabajo Fin de Máster.
