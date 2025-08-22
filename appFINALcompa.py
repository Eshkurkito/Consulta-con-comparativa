import io
from datetime import datetime, timedelta, date
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# =====================================
# Utilidades comunes
# =====================================

def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    required = ["Alojamiento", "Fecha alta", "Fecha entrada", "Fecha salida", "Precio"]
    for col in required:
        if col not in df.columns:
            st.error(f"Falta la columna obligatoria: {col}")
            st.stop()
    df["Fecha alta"] = pd.to_datetime(df["Fecha alta"], errors="coerce")
    df["Fecha entrada"] = pd.to_datetime(df["Fecha entrada"], errors="coerce")
    df["Fecha salida"] = pd.to_datetime(df["Fecha salida"], errors="coerce")
    df["Alojamiento"] = df["Alojamiento"].astype(str).str.strip()
    df["Precio"] = pd.to_numeric(df["Precio"], errors="coerce").fillna(0.0)
    return df

@st.cache_data(show_spinner=False)
def load_excel_from_blobs(file_blobs: List[tuple[str, bytes]]) -> pd.DataFrame:
    """Carga y concatena varios Excel a partir de blobs (nombre, bytes).
    Se cachea por contenido"""
    frames = []
    for name, data in file_blobs:
        try:
            xls = pd.ExcelFile(io.BytesIO(data))
            sheet = (
                "Estado de pagos de las reservas"
                if "Estado de pagos de las reservas" in xls.sheet_names
                else xls.sheet_names[0]
            )
            df = pd.read_excel(xls, sheet_name=sheet)
            df["__source_file__"] = name
            frames.append(df)
        except Exception as e:
            st.error(f"No se pudo leer {name}: {e}")
            st.stop()
    if not frames:
        return pd.DataFrame()
    df_all = pd.concat(frames, ignore_index=True)
    return parse_dates(df_all)

# --- Cálculo vectorizado de KPIs (rápido)

def compute_kpis(
    df_all: pd.DataFrame,
    cutoff: pd.Timestamp,
    period_start: pd.Timestamp,
    period_end: pd.Timestamp,
    inventory_override: Optional[int] = None,
    filter_props: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, dict]:
    # 1) Filtrar por corte y propiedades
    df_cut = df_all[df_all["Fecha alta"] <= cutoff].copy()
    if filter_props:
        df_cut = df_cut[df_cut["Alojamiento"].isin(filter_props)]

    # Quitar filas sin fechas válidas
    df_cut = df_cut.dropna(subset=["Fecha entrada", "Fecha salida"]).copy()

    if df_cut.empty:
        inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
        if inventory_override and inventory_override > 0:
            inv = int(inventory_override)
        days = (period_end - period_start).days + 1
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": inv * days,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    one_day = np.timedelta64(1, 'D')
    start_ns = np.datetime64(pd.to_datetime(period_start))
    end_excl_ns = np.datetime64(pd.to_datetime(period_end) + pd.Timedelta(days=1))  # fin inclusivo

    arr_e = df_cut["Fecha entrada"].values.astype('datetime64[ns]')
    arr_s = df_cut["Fecha salida"].values.astype('datetime64[ns]')

    total_nights = ((arr_s - arr_e) / one_day).astype('int64')
    total_nights = np.clip(total_nights, 0, None)

    ov_start = np.maximum(arr_e, start_ns)
    ov_end = np.minimum(arr_s, end_excl_ns)
    ov_days = ((ov_end - ov_start) / one_day).astype('int64')
    ov_days = np.clip(ov_days, 0, None)

    if ov_days.sum() == 0:
        inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
        if inventory_override and inventory_override > 0:
            inv = int(inventory_override)
        days = (period_end - period_start).days + 1
        total = {
            "noches_ocupadas": 0,
            "noches_disponibles": inv * days,
            "ocupacion_pct": 0.0,
            "ingresos": 0.0,
            "adr": 0.0,
            "revpar": 0.0,
        }
        return pd.DataFrame(columns=["Alojamiento", "Noches ocupadas", "Ingresos", "ADR"]), total

    price = df_cut["Precio"].values.astype('float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        share = np.where(total_nights > 0, ov_days / total_nights, 0.0)
    income = price * share

    props = df_cut["Alojamiento"].astype(str).values
    df_agg = pd.DataFrame({"Alojamiento": props, "Noches": ov_days, "Ingresos": income})
    by_prop = df_agg.groupby("Alojamiento", as_index=False).sum(numeric_only=True)
    by_prop.rename(columns={"Noches": "Noches ocupadas"}, inplace=True)
    by_prop["ADR"] = np.where(by_prop["Noches ocupadas"] > 0, by_prop["Ingresos"] / by_prop["Noches ocupadas"], 0.0)
    by_prop = by_prop.sort_values("Alojamiento")

    noches_ocupadas = int(by_prop["Noches ocupadas"].sum())
    ingresos = float(by_prop["Ingresos"].sum())
    adr = float(ingresos / noches_ocupadas) if noches_ocupadas > 0 else 0.0

    inv = len(set(filter_props)) if filter_props else df_all["Alojamiento"].nunique()
    if inventory_override and inventory_override > 0:
        inv = int(inventory_override)
    days = (period_end - period_start).days + 1
    noches_disponibles = inv * days
    ocupacion_pct = (noches_ocupadas / noches_disponibles * 100) if noches_disponibles > 0 else 0.0
    revpar = ingresos / noches_disponibles if noches_disponibles > 0 else 0.0

    tot = {
        "noches_ocupadas": noches_ocupadas,
        "noches_disponibles": noches_disponibles,
        "ocupacion_pct": ocupacion_pct,
        "ingresos": ingresos,
        "adr": adr,
        "revpar": revpar,
    }

    return by_prop, tot

# =====================================
# App (con archivos persistentes en sesión)
# =====================================

st.set_page_config(page_title="Consultas OTB por corte", layout="wide")
st.title("📅 Consultas OTB – Ocupación, ADR y RevPAR a fecha de corte")
st.caption("Sube archivos una vez y úsalos en cualquiera de los modos.")

# --- Gestor de archivos global ---
with st.sidebar:
    st.header("Archivos de trabajo (persisten en la sesión)")
    files_master = st.file_uploader(
        "Sube uno o varios Excel",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
        key="files_master",
        help="Se admiten múltiples años (2024, 2025…).",
    )
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Usar estos archivos", type="primary"):
            if files_master:
                blobs = [(f.name, f.getvalue()) for f in files_master]
                df_loaded = load_excel_from_blobs(blobs)
                st.session_state["raw_df"] = df_loaded
                st.session_state["file_names"] = [n for n, _ in blobs]
                st.success(f"Cargados {len(blobs)} archivo(s)")
            else:
                st.warning("No seleccionaste archivos.")
    with col_b:
        if st.button("Limpiar archivos"):
            st.session_state.pop("raw_df", None)
            st.session_state.pop("file_names", None)
            st.info("Archivos eliminados de la sesión.")

raw = st.session_state.get("raw_df")
file_names = st.session_state.get("file_names", [])

if raw is not None:
    with st.expander("📂 Archivos cargados"):
        st.write("**Lista:**", file_names)
        st.write(f"**Alojamientos detectados:** {raw['Alojamiento'].nunique()}")
else:
    st.info("Sube archivos en la barra lateral y pulsa **Usar estos archivos** para empezar.")

# -----------------------------
# Menú de modos (independientes)
# -----------------------------
mode = st.sidebar.radio(
    "Modo de consulta",
    ["Consulta normal", "KPIs por meses", "Evolución por fecha de corte"],
)

# Helper para mapear nombres de métricas a columnas
METRIC_MAP = {"Ocupación %": "ocupacion_pct", "ADR (€)": "adr", "RevPAR (€)": "revpar"}

# =============================
# MODO 1: Consulta normal (+ comparación año anterior con inventario propio)
# =============================
if mode == "Consulta normal":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_normal = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="cutoff_normal")
        c1, c2 = st.columns(2)
        with c1:
            start_normal = st.date_input("Inicio del periodo", value=date(2024, 9, 1), key="start_normal")
        with c2:
            end_normal = st.date_input("Fin del periodo", value=date(2024, 9, 30), key="end_normal")
        inv_normal = st.number_input("Sobrescribir inventario (nº alojamientos)", min_value=0, value=0, step=1, key="inv_normal")
        props_normal = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_normal")
        st.markdown("—")
        compare_normal = st.checkbox("Comparar con año anterior (mismo día/mes)", value=False, key="cmp_normal")
        inv_normal_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_normal_prev")

    # Cálculo base
    by_prop_n, total_n = compute_kpis(
        df_all=raw,
        cutoff=pd.to_datetime(cutoff_normal),
        period_start=pd.to_datetime(start_normal),
        period_end=pd.to_datetime(end_normal),
        inventory_override=int(inv_normal) if inv_normal > 0 else None,
        filter_props=props_normal if props_normal else None,
    )

    st.subheader("Resultados totales")
    c1, c2, c3 = st.columns(3)
    c4, c5, c6 = st.columns(3)
    c1.metric("Noches ocupadas", f"{total_n['noches_ocupadas']:,}".replace(",", "."))
    c2.metric("Noches disponibles", f"{total_n['noches_disponibles']:,}".replace(",", "."))
    c3.metric("Ocupación", f"{total_n['ocupacion_pct']:.2f}%")
    c4.metric("Ingresos (€)", f"{total_n['ingresos']:.2f}")
    c5.metric("ADR (€)", f"{total_n['adr']:.2f}")
    c6.metric("RevPAR (€)", f"{total_n['revpar']:.2f}")

    if compare_normal:
        cutoff_cmp = (pd.to_datetime(cutoff_normal) - pd.DateOffset(years=1)).date()
        start_cmp = (pd.to_datetime(start_normal) - pd.DateOffset(years=1)).date()
        end_cmp = (pd.to_datetime(end_normal) - pd.DateOffset(years=1)).date()
        _bp_c, total_cmp = compute_kpis(
            df_all=raw,
            cutoff=pd.to_datetime(cutoff_cmp),
            period_start=pd.to_datetime(start_cmp),
            period_end=pd.to_datetime(end_cmp),
            inventory_override=int(inv_normal_prev) if inv_normal_prev > 0 else None,
            filter_props=props_normal if props_normal else None,
        )
        st.markdown("**Comparativa con año anterior** (corte y periodo -1 año)")
        d1, d2, d3 = st.columns(3)
        d4, d5, d6 = st.columns(3)
        d1.metric("Noches ocupadas (prev)", f"{total_cmp['noches_ocupadas']:,}".replace(",", "."), delta=total_n['noches_ocupadas']-total_cmp['noches_ocupadas'])
        d2.metric("Noches disp. (prev)", f"{total_cmp['noches_disponibles']:,}".replace(",", "."), delta=total_n['noches_disponibles']-total_cmp['noches_disponibles'])
        d3.metric("Ocupación (prev)", f"{total_cmp['ocupacion_pct']:.2f}%", delta=f"{total_n['ocupacion_pct']-total_cmp['ocupacion_pct']:.2f}%")
        d4.metric("Ingresos (prev)", f"{total_cmp['ingresos']:.2f}", delta=f"{total_n['ingresos']-total_cmp['ingresos']:.2f}")
        d5.metric("ADR (prev)", f"{total_cmp['adr']:.2f}", delta=f"{total_n['adr']-total_cmp['adr']:.2f}")
        d6.metric("RevPAR (prev)", f"{total_cmp['revpar']:.2f}", delta=f"{total_n['revpar']-total_cmp['revpar']:.2f}")

    st.divider()
    st.subheader("Detalle por alojamiento")
    if by_prop_n.empty:
        st.warning("Sin noches ocupadas en el periodo a la fecha de corte.")
    else:
        st.dataframe(by_prop_n, use_container_width=True)
        csv = by_prop_n.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar detalle (CSV)", data=csv, file_name="detalle_por_alojamiento.csv", mime="text/csv")

# =============================
# MODO 2: KPIs por meses (línea) + comparación con inventario previo
# =============================
elif mode == "KPIs por meses":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Parámetros")
        cutoff_m = st.date_input("Fecha de corte", value=date(2024, 8, 21), key="cutoff_months")
        props_m = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_months")
        inv_m = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_months")
        inv_m_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_months_prev")
        _min = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).min()
        _max = pd.concat([raw["Fecha entrada"].dropna(), raw["Fecha salida"].dropna()]).max()
        months_options = [str(p) for p in pd.period_range(_min.to_period("M"), _max.to_period("M"), freq="M")] if pd.notna(_min) and pd.notna(_max) else []
        selected_months_m = st.multiselect("Meses a graficar (YYYY-MM)", options=months_options, default=[], key="months_months")
        metric_choice = st.radio("Métrica a graficar", ["Ocupación %", "ADR (€)", "RevPAR (€)"])
        compare_m = st.checkbox("Comparar con año anterior (mismo mes)", value=False, key="cmp_months")

    st.subheader("📈 KPIs por meses (a fecha de corte)")
    if selected_months_m:
        rows_actual = []
        rows_prev = []
        for ym in selected_months_m:
            p = pd.Period(ym, freq="M")
            start_m = p.to_timestamp(how="start")
            end_m = p.to_timestamp(how="end")
            _bp, _tot = compute_kpis(
                df_all=raw,
                cutoff=pd.to_datetime(cutoff_m),
                period_start=start_m,
                period_end=end_m,
                inventory_override=int(inv_m) if inv_m > 0 else None,
                filter_props=props_m if props_m else None,
            )
            rows_actual.append({"Mes": ym, **_tot})

            if compare_m:
                p_prev = p - 12
                start_prev = p_prev.to_timestamp(how="start")
                end_prev = p_prev.to_timestamp(how="end")
                cutoff_prev = pd.to_datetime(cutoff_m) - pd.DateOffset(years=1)
                _bp2, _tot_prev = compute_kpis(
                    df_all=raw,
                    cutoff=cutoff_prev,
                    period_start=start_prev,
                    period_end=end_prev,
                    inventory_override=int(inv_m_prev) if inv_m_prev > 0 else None,
                    filter_props=props_m if props_m else None,
                )
                rows_prev.append({"Mes": ym, **_tot_prev})

        df_actual = pd.DataFrame(rows_actual).sort_values("Mes")
        if not compare_m:
            st.line_chart(df_actual.set_index("Mes")[[METRIC_MAP[metric_choice]]].rename(columns={METRIC_MAP[metric_choice]: metric_choice}), height=280)
            st.dataframe(df_actual[["Mes", "noches_ocupadas", "noches_disponibles", "ocupacion_pct", "adr", "revpar", "ingresos"]].rename(columns={
                "noches_ocupadas": "Noches ocupadas",
                "noches_disponibles": "Noches disponibles",
                "ocupacion_pct": "Ocupación %",
                "adr": "ADR (€)",
                "revpar": "RevPAR (€)",
                "ingresos": "Ingresos (€)"
            }), use_container_width=True)
        else:
            df_prev = pd.DataFrame(rows_prev).sort_values("Mes") if rows_prev else pd.DataFrame()
            plot_df = pd.DataFrame({
                "Actual": df_actual[METRIC_MAP[metric_choice]].values
            }, index=df_actual["Mes"])  # eje X = Mes (string YYYY-MM)
            if not df_prev.empty:
                plot_df["Año anterior"] = df_prev[METRIC_MAP[metric_choice]].values
            st.line_chart(plot_df, height=280)

            table_df = df_actual.merge(df_prev, on="Mes", how="left", suffixes=("", " (prev)")) if not df_prev.empty else df_actual
            rename_map = {
                "noches_ocupadas": "Noches ocupadas",
                "noches_disponibles": "Noches disponibles",
                "ocupacion_pct": "Ocupación %",
                "adr": "ADR (€)",
                "revpar": "RevPAR (€)",
                "ingresos": "Ingresos (€)",
                "noches_ocupadas (prev)": "Noches ocupadas (prev)",
                "noches_disponibles (prev)": "Noches disponibles (prev)",
                "ocupacion_pct (prev)": "Ocupación % (prev)",
                "adr (prev)": "ADR (€) (prev)",
                "revpar (prev)": "RevPAR (€) (prev)",
                "ingresos (prev)": "Ingresos (€) (prev)",
            }
            st.dataframe(table_df.rename(columns=rename_map), use_container_width=True)

        csvm = df_actual.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 Descargar KPIs por mes (CSV)", data=csvm, file_name="kpis_por_mes.csv", mime="text/csv")
    else:
        st.info("Selecciona meses en la barra lateral para ver la gráfica.")

# =============================
# MODO 3: Evolución por fecha de corte + comparación con inventario previo
# =============================
elif mode == "Evolución por fecha de corte":
    if raw is None:
        st.stop()

    with st.sidebar:
        st.header("Rango de corte")
        evo_cut_start = st.date_input("Inicio de corte", value=date(2024, 4, 1), key="evo_cut_start_new")
        evo_cut_end = st.date_input("Fin de corte", value=date(2024, 4, 30), key="evo_cut_end_new")

        st.header("Periodo objetivo")
        evo_target_start = st.date_input("Inicio del periodo", value=date(2024, 9, 1), key="evo_target_start_new")
        evo_target_end = st.date_input("Fin del periodo", value=date(2024, 9, 30), key="evo_target_end_new")

        props_e = st.multiselect("Filtrar alojamientos (opcional)", options=sorted(raw["Alojamiento"].unique()), default=[], key="props_evo")
        inv_e = st.number_input("Inventario actual (opcional)", min_value=0, value=0, step=1, key="inv_evo")
        inv_e_prev = st.number_input("Inventario año anterior (opcional)", min_value=0, value=0, step=1, key="inv_evo_prev")
        metric_choice_e = st.radio("Métrica a graficar", ["Ocupación %", "ADR (€)", "RevPAR (€)"], horizontal=True, key="metric_evo")
        compare_e = st.checkbox("Comparar con año anterior (alineado por día)", value=False, key="cmp_evo")
        run_evo = st.button("Calcular evolución", type="primary", key="btn_evo")

    st.subheader("📉 Evolución de KPIs vs fecha de corte")
    if run_evo:
        cut_start_ts = pd.to_datetime(evo_cut_start)
        cut_end_ts = pd.to_datetime(evo_cut_end)
        if cut_start_ts > cut_end_ts:
            st.error("El inicio del rango de corte no puede ser posterior al fin.")
        else:
            # Serie actual (index = fechas de corte)
            rows_e = []
            for c in pd.date_range(cut_start_ts, cut_end_ts, freq="D"):
                _bp, tot_c = compute_kpis(
                    df_all=raw,
                    cutoff=c,
                    period_start=pd.to_datetime(evo_target_start),
                    period_end=pd.to_datetime(evo_target_end),
                    inventory_override=int(inv_e) if inv_e > 0 else None,
                    filter_props=props_e if props_e else None,
                )
                rows_e.append({"Corte": c.normalize(), **tot_c})
            df_evo = pd.DataFrame(rows_e)

            if df_evo.empty:
                st.info("No hay datos para el rango seleccionado.")
            else:
                key_col = METRIC_MAP[metric_choice_e]
                idx = pd.to_datetime(df_evo["Corte"])  # eje X con fechas reales
                plot_df = pd.DataFrame({"Actual": df_evo[key_col].values}, index=idx)

                if compare_e:
                    # Calcular serie del año anterior y reindexarla a las fechas actuales
                    rows_prev = []
                    cut_start_prev = cut_start_ts - pd.DateOffset(years=1)
                    cut_end_prev = cut_end_ts - pd.DateOffset(years=1)
                    target_start_prev = pd.to_datetime(evo_target_start) - pd.DateOffset(years=1)
                    target_end_prev = pd.to_datetime(evo_target_end) - pd.DateOffset(years=1)
                    prev_dates = list(pd.date_range(cut_start_prev, cut_end_prev, freq="D"))
                    for c in prev_dates:
                        _bp2, tot_c2 = compute_kpis(
                            df_all=raw,
                            cutoff=c,
                            period_start=target_start_prev,
                            period_end=target_end_prev,
                            inventory_override=int(inv_e_prev) if inv_e_prev > 0 else None,
                            filter_props=props_e if props_e else None,
                        )
                        rows_prev.append(tot_c2[key_col])
                    # Reindexar añadiendo +1 año para alinear con las fechas actuales
                    prev_idx_aligned = pd.to_datetime(prev_dates) + pd.DateOffset(years=1)
                    s_prev = pd.Series(rows_prev, index=prev_idx_aligned)
                    plot_df["Año anterior"] = s_prev.reindex(idx).values

                st.line_chart(plot_df, height=300)
                st.dataframe(df_evo[["Corte", "noches_ocupadas", "noches_disponibles", "ocupacion_pct", "adr", "revpar", "ingresos"]].rename(columns={
                    "noches_ocupadas": "Noches ocupadas",
                    "noches_disponibles": "Noches disponibles",
                    "ocupacion_pct": "Ocupación %",
                    "adr": "ADR (€)",
                    "revpar": "RevPAR (€)",
                    "ingresos": "Ingresos (€)"
                }), use_container_width=True)
                csve = df_evo.to_csv(index=False).encode("utf-8-sig")
                st.download_button("📥 Descargar evolución (CSV)", data=csve, file_name="evolucion_kpis.csv", mime="text/csv")
    else:
        st.caption("Configura los parámetros en la barra lateral, luego pulsa **Calcular evolución**.")
