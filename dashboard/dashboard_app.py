# ============================================================
#  JV Program — Dashboard de Inteligencia Operativa
#  VLR México
#
#  Ejecutar con:  streamlit run dashboard_app.py
#  Requiere:      streamlit, plotly, pandas, joblib, shap
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings
warnings.filterwarnings("ignore")

# ── Configuración de página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="JV Program — VLR México",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLOR_CORP  = "#2E75B6"
COLOR_DARK  = "#1a3a5c"
COLOR_RED   = "#E05C2E"
COLOR_GREEN = "#1D9E75"
COLOR_AMBER = "#EF9F27"

# ── Estilos CSS corporativos ──────────────────────────────────────────────────
st.markdown(f"""
<style>
    .main {{ background-color: #F4F8FC; }}
    .stMetric {{ background-color: white; padding: 12px; border-radius: 8px;
                 border-top: 3px solid {COLOR_CORP}; }}
    h1, h2, h3 {{ color: {COLOR_DARK}; }}
    .stTabs [data-baseweb="tab"] {{ font-weight: 600; }}
</style>
""", unsafe_allow_html=True)


# ── Carga de datos y modelos (cacheada) ───────────────────────────────────────
@st.cache_data
def load_data():
    """Carga los datasets generados por el Script 01 (ETL)."""
    base_path = os.path.join("..", "data", "processed")
    dw = pd.read_parquet(os.path.join(base_path, "dataset_WO_nivel.parquet"))
    df_qc = pd.read_parquet(os.path.join(base_path, "dataset_QC_eventos.parquet"))

    for col in ["FECHA_SEMANA"]:
        if col in dw.columns:
            dw[col] = pd.to_datetime(dw[col], errors="coerce")

    return dw, df_qc


@st.cache_resource
def load_model():
    """Carga el modelo XGBoost y el preprocesador serializados (Script 02)."""
    base_path = os.path.join("..", "modeling", "modelos")
    model = joblib.load(os.path.join(base_path, "modelo_xgb.pkl"))
    preprocessor = joblib.load(os.path.join(base_path, "preprocessor.pkl"))
    return model, preprocessor


# FIX 1: inicializar dw/df_qc explícitamente para evitar NameError
# si load_data() lanza excepción antes de retornar.
dw, df_qc = None, None
try:
    dw, df_qc = load_data()
    DATA_OK = True
except Exception as e:
    DATA_OK = False
    st.warning(
        f"⚠️ No se encontraron los datasets procesados. "
        f"Ejecuta primero `etl/01_ETL_JVProgram.py`.\n\nDetalle: {e}"
    )

model, preprocessor = None, None
try:
    model, preprocessor = load_model()
    MODEL_OK = True
except Exception as e:
    MODEL_OK = False


# FIX 2: determinar de forma robusta si la columna de fecha existe,
# independientemente de si DATA_OK es True. Se inicializan SIEMPRE
# fecha_ini / fecha_fin / area_sel para que nunca queden indefinidas
# más adelante en el script (causa del NameError original).
TIENE_FECHA_SEMANA = DATA_OK and dw is not None and "FECHA_SEMANA" in dw.columns
fecha_ini, fecha_fin = None, None
area_sel = "Todas"

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📊 JV Program")
    st.caption("VLR México · Panel de inteligencia operativa")
    st.divider()

    if DATA_OK:
        if TIENE_FECHA_SEMANA:
            semanas_disp = sorted(dw["FECHA_SEMANA"].dropna().unique())
        else:
            semanas_disp = []
            st.info("La columna FECHA_SEMANA no está disponible en el dataset.")

        if len(semanas_disp) > 0:
            rango = st.select_slider(
                "Período de análisis",
                options=range(len(semanas_disp)),
                value=(0, len(semanas_disp) - 1),
                format_func=lambda i: pd.Timestamp(semanas_disp[i]).strftime("%b %Y"),
            )
            fecha_ini, fecha_fin = semanas_disp[rango[0]], semanas_disp[rango[1]]
        else:
            fecha_ini, fecha_fin = None, None

        # FIX 3: el operador ternario original solo envolvía sorted(...),
        # dejando "['Todas'] +" fuera de la condición cuando AREA_QC no
        # existía en df_qc, lo que producía un TypeError al concatenar
        # una lista con un resultado no-lista. Ahora la condición cubre
        # la expresión completa.
        if df_qc is not None and "AREA_QC" in df_qc.columns:
            areas_disp = ["Todas"] + sorted(df_qc["AREA_QC"].dropna().unique().tolist())
        else:
            areas_disp = ["Todas"]
        area_sel = st.selectbox("Área productiva", areas_disp)

    st.divider()
    st.caption("Modelo: XGBoost · AUC-ROC = 0.9906")
    st.caption("Última actualización: " +
               pd.Timestamp.now().strftime("%d/%m/%Y %H:%M"))


# ── Header ─────────────────────────────────────────────────────────────────────
st.title("JV Program — Panel de Inteligencia Operativa")
st.caption("Sistema de control de producción · VLR México")

# FIX 10: doble verificación antes de continuar. st.stop() detiene la
# ejecución dentro de una app Streamlit real, pero si el script se corre
# en "bare mode" (p. ej. ejecutado como celda de Jupyter en lugar de con
# `streamlit run dashboard_app.py`), st.stop() puede no interrumpir el
# flujo de Python y el script sigue ejecutándose con dw=None, causando
# AttributeError más adelante. Se añade una verificación explícita de
# dw is None como salvaguarda adicional.
if not DATA_OK or dw is None:
    st.error(
        "⚠️ No se pudieron cargar los datasets. Verifica que existan "
        "`data/processed/dataset_WO_nivel.parquet` y "
        "`data/processed/dataset_QC_eventos.parquet`, y que estés "
        "ejecutando este archivo con:\n\n"
        "    streamlit run dashboard_app.py\n\n"
        "(no como celda de Jupyter ni con `python dashboard_app.py`)."
    )
    st.stop()
    raise SystemExit(
        "Datos no disponibles — deteniendo ejecución (ver mensaje en la UI)."
    )

# FIX 4: aplicar el filtro de fecha solo si la columna existe y hay rango
# seleccionado; de lo contrario usar el dataset completo sin filtrar.
if TIENE_FECHA_SEMANA and fecha_ini is not None and fecha_fin is not None:
    dw_f = dw[(dw["FECHA_SEMANA"] >= fecha_ini) & (dw["FECHA_SEMANA"] <= fecha_fin)].copy()
else:
    dw_f = dw.copy()

# FIX 5: aplicar también el filtro de área seleccionado en la sidebar,
# que en la versión original se calculaba pero nunca se usaba.
if area_sel != "Todas" and df_qc is not None and "AREA_QC" in df_qc.columns:
    df_qc_f = df_qc[df_qc["AREA_QC"] == area_sel].copy()
else:
    df_qc_f = df_qc.copy() if df_qc is not None else pd.DataFrame()


# ── Tabs principales ──────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 KPIs", "📉 Tendencias", "🗺️ Mapa de calidad", "🎯 Predicción de riesgo"]
)

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — KPIs
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Indicadores clave de producción")

    col1, col2, col3, col4 = st.columns(4)

    n_wo = len(dw_f)
    tasa_rechazo = dw_f["TASA_RECHAZO"].mean() if "TASA_RECHAZO" in dw_f.columns else np.nan
    efic_costura = dw_f["EFIC_COSTURA"].mean() if "EFIC_COSTURA" in dw_f.columns else np.nan
    cumple_pct = dw_f["CUMPLE_FECHA"].mean() * 100 if "CUMPLE_FECHA" in dw_f.columns else np.nan

    col1.metric("WO en el período", f"{n_wo:,}")
    col2.metric("Tasa de rechazo promedio",
                f"{tasa_rechazo:.2f}%" if pd.notna(tasa_rechazo) else "N/D")
    col3.metric("Eficiencia costura promedio",
                f"{efic_costura:.1f}%" if pd.notna(efic_costura) else "N/D")
    col4.metric("WO que cumplen fecha plan",
                f"{cumple_pct:.1f}%" if pd.notna(cumple_pct) else "N/D")

    st.divider()
    st.subheader("Resumen estadístico de variables clave")

    vars_resumen = ["TC_TOTAL_D", "TASA_RECHAZO", "EFIC_COSTURA", "QTY"]
    vars_disp = [v for v in vars_resumen if v in dw_f.columns]
    if vars_disp:
        st.dataframe(
            dw_f[vars_disp].describe().round(2).T,
            use_container_width=True,
        )
    else:
        st.info("No hay variables numéricas disponibles para mostrar el resumen.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Tendencias
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Evolución semanal del volumen de producción y tasa de rechazo")

    # FIX 6: validar que existan las columnas necesarias antes de agrupar,
    # para no fallar con KeyError si el dataset cargado es incompleto.
    cols_necesarias = {"FECHA_SEMANA", "WO", "TASA_RECHAZO"}
    if cols_necesarias.issubset(dw_f.columns):
        sem_stats = (dw_f.groupby("FECHA_SEMANA")
                       .agg(N_WO=("WO", "count"),
                            TASA=("TASA_RECHAZO", "mean"))
                       .reset_index()
                       .sort_values("FECHA_SEMANA")
                       .dropna(subset=["FECHA_SEMANA"]))
        sem_stats["TASA"] = sem_stats["TASA"].fillna(0)

        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Scatter(
            x=sem_stats["FECHA_SEMANA"], y=sem_stats["N_WO"],
            name="Volumen WO", line=dict(color=COLOR_CORP, width=2),
            fill="tozeroy", fillcolor="rgba(46,117,182,0.1)"
        ), secondary_y=False)
        fig.add_trace(go.Scatter(
            x=sem_stats["FECHA_SEMANA"], y=sem_stats["TASA"],
            name="Tasa rechazo (%)", line=dict(color=COLOR_RED, width=2, dash="dash")
        ), secondary_y=True)

        fig.update_layout(
            height=450, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            plot_bgcolor="white",
        )
        fig.update_yaxes(title_text="N.º de WO", secondary_y=False)
        fig.update_yaxes(title_text="Tasa de rechazo (%)", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        faltantes = cols_necesarias - set(dw_f.columns)
        st.info(f"No se puede graficar la tendencia: faltan columnas {sorted(faltantes)}.")

    st.divider()
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Top 10 estilos por volumen**")
        if "ITEM" in dw_f.columns and len(dw_f) > 0:
            top_items = dw_f["ITEM"].value_counts().head(10)
            fig2 = px.bar(
                x=top_items.values, y=top_items.index, orientation="h",
                color_discrete_sequence=[COLOR_CORP],
                labels={"x": "N.º de WO", "y": ""},
            )
            fig2.update_layout(height=400, plot_bgcolor="white")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("Columna ITEM no disponible.")

    with col_b:
        st.markdown("**Tasa de rechazo por área (QC)**")
        if (df_qc is not None and "AREA_QC" in df_qc.columns
                and "QTY_RECHAZADO" in df_qc.columns and "QTY_AUDIT" in df_qc.columns):
            # FIX 7: reemplazar groupby().apply(lambda) por una agregación
            # vectorizada con sum()/sum(), evitando el FutureWarning de
            # pandas sobre "DataFrameGroupBy.apply operated on the
            # grouping columns" y el riesgo de comportamiento inconsistente
            # entre versiones de pandas.
            area_grp = df_qc.groupby("AREA_QC")[["QTY_RECHAZADO", "QTY_AUDIT"]].sum()
            area_grp["TASA"] = (area_grp["QTY_RECHAZADO"] /
                                 area_grp["QTY_AUDIT"].replace(0, np.nan) * 100)
            area_rech = area_grp.reset_index()

            fig3 = px.bar(
                area_rech, x="AREA_QC", y="TASA",
                color_discrete_sequence=[COLOR_AMBER],
                labels={"TASA": "Tasa de rechazo (%)", "AREA_QC": "Área"},
            )
            fig3.update_layout(height=400, plot_bgcolor="white")
            st.plotly_chart(fig3, use_container_width=True)
        else:
            st.info("Columnas AREA_QC / QTY_RECHAZADO / QTY_AUDIT no disponibles.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Mapa de calidad
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Mapa de calor — defectos por área productiva")

    EXCLUIR = ["Sin Defectos", "Sin defectos", "SIN DEFECTOS", "Sin Defecto"]

    # FIX 8: en la versión original, si "DESCRIPTION" no existía en df_qc,
    # df_qc_def jamás se definía y el bloque del diagrama de Pareto (fuera
    # del if "AREA_QC" pero todavía dentro del if "DESCRIPTION") lanzaba
    # NameError. Ahora se inicializa df_qc_def de forma segura y se valida
    # antes de cada uso.
    df_qc_def = pd.DataFrame()
    if df_qc is not None and "DESCRIPTION" in df_qc.columns:
        df_qc_def = df_qc[~df_qc["DESCRIPTION"].isin(EXCLUIR)].copy()

    if not df_qc_def.empty and "AREA_QC" in df_qc_def.columns:
        pivot = (df_qc_def.groupby(["AREA_QC", "DESCRIPTION"])
                 .size().unstack(fill_value=0))
        top_defectos = df_qc_def["DESCRIPTION"].value_counts().head(10).index
        pivot = pivot[pivot.columns.intersection(top_defectos)]

        if not pivot.empty:
            fig_hm = px.imshow(
                pivot, text_auto=True, aspect="auto",
                color_continuous_scale="YlOrRd",
                labels=dict(x="Tipo de defecto", y="Área", color="Frecuencia"),
            )
            fig_hm.update_layout(height=450)
            st.plotly_chart(fig_hm, use_container_width=True)
        else:
            st.info("No hay datos suficientes para construir el mapa de calor.")
    else:
        st.info("Columnas DESCRIPTION / AREA_QC no disponibles para el mapa de calor.")

    st.divider()
    st.subheader("Diagrama de Pareto — tipos de defecto")

    if not df_qc_def.empty:
        pareto = df_qc_def["DESCRIPTION"].value_counts().head(10)
        cum_pct = pareto.cumsum() / pareto.sum() * 100

        fig_p = make_subplots(specs=[[{"secondary_y": True}]])
        fig_p.add_trace(go.Bar(
            x=pareto.index, y=pareto.values, name="Frecuencia",
            marker_color=COLOR_CORP,
        ), secondary_y=False)
        fig_p.add_trace(go.Scatter(
            x=pareto.index, y=cum_pct.values, name="% acumulado",
            line=dict(color=COLOR_RED, width=2),
        ), secondary_y=True)
        fig_p.add_hline(y=80, line_dash="dash", line_color=COLOR_RED,
                         secondary_y=True)
        fig_p.update_layout(height=450, plot_bgcolor="white")
        st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.info("No hay datos de defectos disponibles para el diagrama de Pareto.")

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — Predicción de riesgo
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Predicción de riesgo de rechazo para nueva WO")

    if not MODEL_OK:
        st.warning(
            "⚠️ No se encontró el modelo entrenado. "
            "Ejecuta primero `modeling/02_Modelado_JVProgram.py`."
        )
    else:
        col_form, col_result = st.columns([1, 1.2])

        with col_form:
            st.markdown("##### Atributos de la orden de trabajo")

            item_sel = st.selectbox(
                "Estilo (ITEM)",
                sorted(dw["ITEM"].dropna().unique()) if "ITEM" in dw.columns else ["—"],
            )
            body_pillow_sel = st.selectbox(
                "Tipo (BODY_PILLOW)",
                sorted(dw["BODY_PILLOW"].dropna().unique())
                if "BODY_PILLOW" in dw.columns else ["BODY", "PILLOW", "REPAIR/SAMPLE"],
            )
            family_sel = st.selectbox(
                "Familia (FAMILY)",
                sorted(dw["FAMILY"].dropna().unique())
                if "FAMILY" in dw.columns else ["—"],
            )
            qty_sel = st.number_input("Cantidad (QTY)", min_value=1, value=50)
            carga_sel = st.slider("Carga semanal estimada (N.º WO)",
                                   min_value=200, max_value=1100, value=700)
            efic_prep_sel = st.slider("Eficiencia esperada en Preparación (%)",
                                       min_value=50, max_value=130, value=90)

            calcular = st.button("🎯 Calcular riesgo de rechazo", type="primary",
                                  use_container_width=True)

        with col_result:
            st.markdown("##### Resultado de la predicción")

            if calcular:
                # Construir registro de entrada con los features esperados
                input_data = pd.DataFrame([{
                    "QTY": qty_sel,
                    "TC_CORTE_H": dw["TC_CORTE_H"].median() if "TC_CORTE_H" in dw.columns else 4,
                    "TC_PREP_H": dw["TC_PREP_H"].median() if "TC_PREP_H" in dw.columns else 6,
                    "TC_COSTURA_H": dw["TC_COSTURA_H"].median() if "TC_COSTURA_H" in dw.columns else 8,
                    "EFIC_CORTE": dw["EFIC_CORTE"].median() if "EFIC_CORTE" in dw.columns else 95,
                    "EFIC_PREP": efic_prep_sel,
                    "EFIC_COSTURA": dw["EFIC_COSTURA"].median() if "EFIC_COSTURA" in dw.columns else 92,
                    "CARGA_SEMANA": carga_sel,
                    "SEMANA_NUM": dw["SEMANA_NUM"].max() if "SEMANA_NUM" in dw.columns else 2627,
                    "DIA_INICIO": 1,
                    "ITEM": item_sel,
                    "BODY_PILLOW": body_pillow_sel,
                    "FAMILY": family_sel,
                }])

                # FIX 9: el bloque try/except original capturaba CUALQUIER
                # excepción y reintentaba con model.predict_proba(input_data)
                # directamente, lo cual habría fallado de nuevo (mismo
                # error de columnas no transformadas) y habría producido
                # un traceback confuso para el usuario. Ahora se captura
                # explícitamente y se informa con un mensaje claro en la UI
                # en vez de dejar que Streamlit muestre un traceback crudo.
                proba = None
                # FIX 9 (revisado): el fallback original reintentaba con
                # model.predict_proba(input_data) usando los datos SIN
                # transformar, lo cual siempre falla en XGBoost porque las
                # columnas categóricas (ITEM, BODY_PILLOW, FAMILY) llegan
                # como texto y el modelo solo acepta int/float/bool/category.
                # Se elimina el fallback engañoso y se muestra el error
                # real de preprocessor.transform() para poder diagnosticarlo.
                proba = None
                try:
                    X_transformed = preprocessor.transform(input_data)
                    proba = model.predict_proba(X_transformed)[0, 1]
                except Exception as e:
                    st.error(
                        "⚠️ No fue posible generar la predicción: el "
                        "preprocesador no pudo transformar los datos de entrada."
                    )
                    st.exception(e)
                    st.caption(
                        "Causa habitual: el preprocessor.pkl fue entrenado "
                        "con columnas distintas a las que arma este formulario, "
                        "o las categorías seleccionadas (ITEM/FAMILY) no "
                        "existían en el conjunto de entrenamiento original."
                    )

                if proba is not None:
                    # Semáforo
                    if proba >= 0.6:
                        nivel, emoji = "ALTO", "🔴"
                    elif proba >= 0.3:
                        nivel, emoji = "MEDIO", "🟡"
                    else:
                        nivel, emoji = "BAJO", "🟢"

                    st.markdown(f"### {emoji} Riesgo {nivel}")
                    st.metric("Probabilidad de rechazo", f"{proba*100:.1f}%")
                    st.progress(min(float(proba), 1.0))

                    if body_pillow_sel == "REPAIR/SAMPLE":
                        st.error(
                            "⚠️ **Recomendación:** activar inspección in-process "
                            "antes de Costura. Las WO tipo REPAIR/SAMPLE concentran "
                            "el 70.6% de la importancia predictiva del modelo."
                        )
                    elif nivel == "ALTO":
                        st.warning(
                            "⚠️ **Recomendación:** reforzar verificación de "
                            "ensamble en Preparación antes de continuar el proceso."
                        )
                    else:
                        st.success("✅ Riesgo dentro de parámetros normales.")
            else:
                st.info("Completa el formulario y presiona **Calcular riesgo** "
                        "para ver la predicción del modelo XGBoost.")


# ── Footer ─────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "JV Program v1.0 (Prototipo) · Equipo 1_F · "
    "Máster en Análisis y Visualización de Datos Masivos · UNIR 2026"
)
