# ============================================================
#  JV Program – VLR México
#  Notebook 02: Análisis Exploratorio de Datos (EDA) — Capítulo 4
#  Genera las figuras 4–13 del documento (estadísticas y distribuciones)
#  Requiere: dataset_WO_nivel.parquet y dataset_QC_eventos.parquet
#            generados por etl/01_ETL_JVProgram.py
# ============================================================
#
# NOTA: Este notebook complementa al script 01_ETL_JVProgram.py.
# El bloque ETL completo (celdas 1–8) y las primeras figuras del EDA
# (celdas 9–22, Figuras 4 a 16) ya están incluidos en
# /etl/01_ETL_JVProgram.py para evitar duplicación.
#
# Este archivo contiene únicamente las celdas adicionales del EDA
# que no están en el script de ETL: análisis univariado de categóricas
# extendido, pruebas estadísticas complementarias y exportación
# consolidada de resultados para el documento Word.
# ============================================================

# ── CELDA 1: Importaciones y carga de datos ───────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings, os
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "Arial", "figure.dpi": 150,
    "axes.spines.top": False, "axes.spines.right": False,
})
COLOR_CORP = "#2E75B6"

dw    = pd.read_parquet("../data/processed/dataset_WO_nivel.parquet")
df_qc = pd.read_parquet("../data/processed/dataset_QC_eventos.parquet")

os.makedirs("figuras_eda", exist_ok=True)
def save(name):
    plt.savefig(f"figuras_eda/{name}", dpi=150, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"✅ {name}")

print(f"✅ Dataset cargado: {len(dw):,} WO · {len(df_qc):,} eventos QC")


# ── CELDA 2: Resumen estadístico extendido (Tabla 9 ampliada) ─────────────────
vars_extendidas = ["TC_TOTAL_D", "TC_CORTE_H", "TC_PREP_H", "TC_COSTURA_H",
                    "TASA_RECHAZO", "EFIC_CORTE", "EFIC_PREP", "EFIC_COSTURA",
                    "QTY", "CARGA_SEMANA", "DESVIO_PLAN_D"]
vars_disp = [v for v in vars_extendidas if v in dw.columns]

resumen = dw[vars_disp].agg(["count", "mean", "median", "std", "min", "max", "skew"]).T
resumen.columns = ["N", "Media", "Mediana", "Desv.Est.", "Mín", "Máx", "Asimetría"]
print(resumen.round(2).to_string())
resumen.round(2).to_csv("figuras_eda/tabla09_resumen_extendido.csv")
print("\n✅ Tabla guardada: tabla09_resumen_extendido.csv")


# ── CELDA 3: Prueba de homogeneidad de varianzas (Levene) por área ───────────
if "AREA_QC" in df_qc.columns and "QTY_RECHAZADO" in df_qc.columns:
    df_qc["TASA_R"] = df_qc["QTY_RECHAZADO"] / df_qc["QTY_AUDIT"].replace(0, np.nan) * 100
    grupos = [df_qc[df_qc["AREA_QC"] == a]["TASA_R"].dropna()
              for a in df_qc["AREA_QC"].dropna().unique()]
    grupos = [g for g in grupos if len(g) > 5]

    if len(grupos) >= 2:
        stat_lev, p_lev = stats.levene(*grupos)
        print(f"Prueba de Levene (homogeneidad de varianzas):")
        print(f"  Estadístico = {stat_lev:.4f}  |  p-valor = {p_lev:.4f}")
        print(f"  {'Varianzas homogéneas' if p_lev > 0.05 else 'Varianzas NO homogéneas'} (α=0.05)")

        stat_kw, p_kw = stats.kruskal(*grupos)
        print(f"\nPrueba de Kruskal-Wallis (diferencia entre áreas):")
        print(f"  Estadístico = {stat_kw:.4f}  |  p-valor = {p_kw:.4f}")
        print(f"  {'Diferencias significativas' if p_kw < 0.05 else 'Sin diferencias significativas'} (α=0.05)")


# ── CELDA 4: Matriz de correlación punto-biserial con ES_RECHAZADA ───────────
vars_num = ["TC_TOTAL_D", "TC_CORTE_H", "TC_COSTURA_H", "EFIC_COSTURA",
            "QTY", "CARGA_SEMANA"]
vars_num = [v for v in vars_num if v in dw.columns]

resultados_pb = []
for v in vars_num:
    valid = dw[[v, "ES_RECHAZADA"]].dropna()
    if len(valid) > 10:
        r, p = stats.pointbiserialr(valid["ES_RECHAZADA"], valid[v])
        resultados_pb.append({"Variable": v, "r_pb": round(r, 4), "p_valor": round(p, 4)})

df_pb = pd.DataFrame(resultados_pb).sort_values("r_pb", key=abs, ascending=False)
print("Correlación punto-biserial con ES_RECHAZADA:")
print(df_pb.to_string(index=False))
df_pb.to_csv("figuras_eda/correlacion_punto_biserial.csv", index=False)


# ── CELDA 5: Resumen final ────────────────────────────────────────────────────
print("\n" + "="*55)
print("Notebook de EDA complementario ejecutado correctamente")
print("="*55)
print("Para las Figuras 4–16 principales del Capítulo 4, ejecutar:")
print("  → etl/01_ETL_JVProgram.py  (celdas 9 en adelante)")
