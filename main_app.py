import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Baguio City Dengue Forecast Dashboard",
    page_icon="🦟",
    layout="wide",
)

# =========================
# CONFIG
# =========================
ARTIFACTS_DIR = Path("artifacts")

DEFAULT_FEATURE_COLS = [
    "rainfall", "relative_humidity", "temp_mid",
    "cases_lag_1", "cases_lag_2", "cases_lag_3",
    "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
    "relative_humidity_lag_1", "relative_humidity_lag_2", "relative_humidity_lag_3",
    "temp_mid_lag_1", "temp_mid_lag_2", "temp_mid_lag_3",
    "cases_roll3_mean", "cases_roll3_max",
    "month_sin", "month_cos",
]

ARTIFACT_SPECS = {
    "monthly":                  ["monthly_modeling_dataset.csv"],
    "model_comparison":         ["model_comparison.csv"],
    "auc_df":                   ["model_auc.csv"],
    "feature_importance":       ["feature_importance.csv"],
    "feature_sensitivity":      ["feature_sensitivity.csv"],
    "forecast":                 ["forecast_5yr.csv"],
    "barangay_monthly":         ["barangay_monthly.csv"],
    "top_barangay_monthly":     ["top_barangay_monthly.csv"],
    "top3_barangays_yearly":    ["top3_barangays_yearly.csv"],
    "top3_barangays_overall":   ["top3_barangays_overall.csv"],
    "test_predictions":         ["test_predictions.csv"],
    "climate_case_correlation": ["climate_case_correlation.csv"],
    "month_profile":            ["month_profile.csv"],
    "forecast_top3_barangays":  ["forecast_top3_barangays.csv"],
    "barangay_risk_profile":    ["barangay_risk_profile.csv"],
}

DATE_COLUMNS = {
    "monthly", "forecast", "barangay_monthly", "top_barangay_monthly",
    "test_predictions", "forecast_top3_barangays",
}

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Baguio City Dengue Forecast Dashboard")
st.caption("Interactive web-based dashboard for predicting dengue outbreaks in Baguio City.")


# =========================
# LOADING HELPERS
# =========================
def safe_read_csv_from_names(file_names):
    for name in file_names:
        path = ARTIFACTS_DIR / name
        if path.exists():
            return pd.read_csv(path)
    return None


def safe_read_json(path: Path):
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def safe_load_model(path: Path):
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data(show_spinner=False)
def load_artifacts():
    loaded = {key: safe_read_csv_from_names(names) for key, names in ARTIFACT_SPECS.items()}
    loaded["meta"] = safe_read_json(ARTIFACTS_DIR / "meta.json")
    return loaded


artifacts = load_artifacts()
model = safe_load_model(ARTIFACTS_DIR / "best_model.joblib")


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Dashboard Files")
st.sidebar.write(
    "Place your exported Colab files inside an `artifacts/` folder, or upload them manually below."
)

meta = artifacts.get("meta")
if meta:
    st.sidebar.success(f"Best Model: {meta.get('best_model', 'Unknown')}")
    threshold_val = meta.get("outbreak_threshold_cases", "N/A")
    if isinstance(threshold_val, (int, float)):
        st.sidebar.info(f"Outbreak Threshold: {threshold_val:.2f} cases")
    else:
        st.sidebar.info(f"Outbreak Threshold: {threshold_val}")
else:
    st.sidebar.warning("meta.json not found.")

with st.sidebar.expander("Manual file upload", expanded=False):
    uploaded_files = {}
    uploaded_files["monthly"]                  = st.file_uploader("monthly_modeling_dataset.csv", type=["csv"])
    uploaded_files["model_comparison"]         = st.file_uploader("model_comparison.csv", type=["csv"])
    uploaded_files["auc_df"]                   = st.file_uploader("model_auc.csv", type=["csv"])
    uploaded_files["feature_importance"]       = st.file_uploader("feature_importance.csv", type=["csv"])
    uploaded_files["feature_sensitivity"]      = st.file_uploader("feature_sensitivity.csv", type=["csv"])
    uploaded_files["forecast"]                 = st.file_uploader("forecast_5yr.csv", type=["csv"])
    uploaded_files["barangay_monthly"]         = st.file_uploader("barangay_monthly.csv", type=["csv"])
    uploaded_files["top_barangay_monthly"]     = st.file_uploader("top_barangay_monthly.csv", type=["csv"])
    uploaded_files["top3_barangays_yearly"]    = st.file_uploader("top3_barangays_yearly.csv", type=["csv"])
    uploaded_files["top3_barangays_overall"]   = st.file_uploader("top3_barangays_overall.csv", type=["csv"])
    uploaded_files["test_predictions"]         = st.file_uploader("test_predictions.csv", type=["csv"])
    uploaded_files["climate_case_correlation"] = st.file_uploader("climate_case_correlation.csv", type=["csv"])
    uploaded_files["month_profile"]            = st.file_uploader("month_profile.csv", type=["csv"])
    uploaded_files["forecast_top3_barangays"]  = st.file_uploader("forecast_top3_barangays.csv", type=["csv"])
    uploaded_files["barangay_risk_profile"]    = st.file_uploader("barangay_risk_profile.csv", type=["csv"])
    uploaded_meta  = st.file_uploader("meta.json", type=["json"])
    uploaded_model = st.file_uploader("best_model.joblib", type=["joblib", "pkl"])

for key, uploaded in uploaded_files.items():
    if uploaded is not None:
        artifacts[key] = pd.read_csv(uploaded)

if uploaded_meta is not None:
    meta = json.load(uploaded_meta)
    artifacts["meta"] = meta

if uploaded_model is not None:
    model = joblib.load(uploaded_model)

monthly                  = artifacts.get("monthly")
model_comparison         = artifacts.get("model_comparison")
auc_df                   = artifacts.get("auc_df")
feature_importance       = artifacts.get("feature_importance")
feature_sensitivity      = artifacts.get("feature_sensitivity")
forecast                 = artifacts.get("forecast")
barangay_monthly         = artifacts.get("barangay_monthly")
top_barangay_monthly     = artifacts.get("top_barangay_monthly")
top3_barangays_yearly    = artifacts.get("top3_barangays_yearly")
top3_barangays_overall   = artifacts.get("top3_barangays_overall")
test_predictions         = artifacts.get("test_predictions")
climate_case_correlation = artifacts.get("climate_case_correlation")
month_profile            = artifacts.get("month_profile")
forecast_top3_barangays  = artifacts.get("forecast_top3_barangays")
barangay_risk_profile    = artifacts.get("barangay_risk_profile")

if monthly is None or monthly.empty:
    st.error(
        "monthly_modeling_dataset.csv is required. Export it from Colab and place it in the artifacts/ folder or upload it in the sidebar."
    )
    st.stop()

# Parse Date columns
for key in DATE_COLUMNS:
    df_obj = locals().get(key)
    if df_obj is not None and "Date" in df_obj.columns:
        df_obj = df_obj.copy()
        df_obj["Date"] = pd.to_datetime(df_obj["Date"], errors="coerce")
        locals()[key] = df_obj


# =========================
# HELPERS
# =========================
def month_name_from_number(month_num):
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    try:
        return names.get(int(month_num), str(month_num))
    except Exception:
        return str(month_num)


def safe_metric_value(value, decimals=2):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


def round_display_columns(df, columns, decimals=2):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
    return df


def display_clean_barangay_columns(df):
    if df is None:
        return None
    df = df.copy()
    rename_map = {
        "Top_Barangay": "Barangay",
        "Top_Barangay_Cases": "Barangay Cases",
        "rank_within_year": "Rank Within Year",
        "rank": "Rank",
        "Barangay_cases": "Dengue Cases",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def get_profile_value(month_num, col_name, month_profile_df, fallback_df=None, default=0.0):
    if month_profile_df is not None and not month_profile_df.empty and "Month" in month_profile_df.columns:
        subset = month_profile_df[pd.to_numeric(month_profile_df["Month"], errors="coerce") == int(month_num)]
        if not subset.empty and col_name in subset.columns:
            value = pd.to_numeric(pd.Series([subset.iloc[0][col_name]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)
    if fallback_df is not None and col_name in fallback_df.columns:
        series = pd.to_numeric(fallback_df[col_name], errors="coerce").dropna()
        if len(series) > 0:
            return float(series.mean())
    return float(default)


def get_reasonable_range(df, col_name, fallback_min=0.0, fallback_max=100.0):
    if df is not None and col_name in df.columns:
        series = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if len(series) > 0:
            vmin, vmax = float(series.min()), float(series.max())
            if vmin == vmax:
                vmax = vmin + 1.0
            return vmin, vmax
    return fallback_min, fallback_max


def get_forecast_row(forecast_df, year_num, month_num):
    if forecast_df is None or forecast_df.empty or not {"Year", "Month"}.issubset(forecast_df.columns):
        return None
    subset = forecast_df[
        (pd.to_numeric(forecast_df["Year"], errors="coerce") == int(year_num)) &
        (pd.to_numeric(forecast_df["Month"], errors="coerce") == int(month_num))
    ]
    return subset.iloc[0] if not subset.empty else None


def previous_month_numbers(year_num, month_num):
    periods = []
    y, m = int(year_num), int(month_num)
    for _ in range(3):
        m -= 1
        if m == 0:
            m = 12
            y -= 1
        periods.append((y, m))
    return periods


def build_live_prediction_features(
    year_num, month_num,
    rainfall_now, humidity_now, temp_now,
    cases_lag_1, cases_lag_2, cases_lag_3,
    month_profile_df, monthly_df, forecast_row=None,
):
    if forecast_row is not None:
        def _fv(key):
            return float(pd.to_numeric(pd.Series([forecast_row.get(key, np.nan)]), errors="coerce").fillna(0.0).iloc[0])
        rainfall_lag_1, rainfall_lag_2, rainfall_lag_3 = _fv("rainfall_lag_1"), _fv("rainfall_lag_2"), _fv("rainfall_lag_3")
        rh_lag_1, rh_lag_2, rh_lag_3 = _fv("relative_humidity_lag_1"), _fv("relative_humidity_lag_2"), _fv("relative_humidity_lag_3")
        temp_lag_1, temp_lag_2, temp_lag_3 = _fv("temp_mid_lag_1"), _fv("temp_mid_lag_2"), _fv("temp_mid_lag_3")
    else:
        prev = previous_month_numbers(year_num, month_num)
        (_, p1), (_, p2), (_, p3) = prev
        rainfall_lag_1 = get_profile_value(p1, "rainfall", month_profile_df, monthly_df, 0.0)
        rainfall_lag_2 = get_profile_value(p2, "rainfall", month_profile_df, monthly_df, 0.0)
        rainfall_lag_3 = get_profile_value(p3, "rainfall", month_profile_df, monthly_df, 0.0)
        rh_lag_1 = get_profile_value(p1, "relative_humidity", month_profile_df, monthly_df, 0.0)
        rh_lag_2 = get_profile_value(p2, "relative_humidity", month_profile_df, monthly_df, 0.0)
        rh_lag_3 = get_profile_value(p3, "relative_humidity", month_profile_df, monthly_df, 0.0)
        temp_lag_1 = get_profile_value(p1, "temp_mid", month_profile_df, monthly_df, 0.0)
        temp_lag_2 = get_profile_value(p2, "temp_mid", month_profile_df, monthly_df, 0.0)
        temp_lag_3 = get_profile_value(p3, "temp_mid", month_profile_df, monthly_df, 0.0)

    return {
        "rainfall":                 float(rainfall_now),
        "relative_humidity":        float(humidity_now),
        "temp_mid":                 float(temp_now),
        "cases_lag_1":              float(cases_lag_1),
        "cases_lag_2":              float(cases_lag_2),
        "cases_lag_3":              float(cases_lag_3),
        "rainfall_lag_1":           float(rainfall_lag_1),
        "rainfall_lag_2":           float(rainfall_lag_2),
        "rainfall_lag_3":           float(rainfall_lag_3),
        "relative_humidity_lag_1":  float(rh_lag_1),
        "relative_humidity_lag_2":  float(rh_lag_2),
        "relative_humidity_lag_3":  float(rh_lag_3),
        "temp_mid_lag_1":           float(temp_lag_1),
        "temp_mid_lag_2":           float(temp_lag_2),
        "temp_mid_lag_3":           float(temp_lag_3),
        "cases_roll3_mean":         float(np.mean([cases_lag_1, cases_lag_2, cases_lag_3])),
        "cases_roll3_max":          float(np.max([cases_lag_1, cases_lag_2, cases_lag_3])),
        "month_sin":                float(np.sin(2 * np.pi * int(month_num) / 12.0)),
        "month_cos":                float(np.cos(2 * np.pi * int(month_num) / 12.0)),
    }


def outbreak_label_from_binary(x):
    try:
        return "Outbreak" if int(x) == 1 else "Non-outbreak"
    except Exception:
        return "Unknown"


def normalize_model_comparison(model_df, auc_table):
    """Rename model_comparison columns to dashboard-expected names and merge AUC."""
    if model_df is None or model_df.empty:
        return model_df
    model_df = model_df.copy()
    col_renames = {
        "Model":                  "model",
        "F1 Score":               "f1_score",
        "Precision":              "precision",
        "Recall":                 "recall",
        "Reliability (Brier)":    "brier",
        "AUC (Supplementary)":   "auc",
    }
    model_df = model_df.rename(columns={k: v for k, v in col_renames.items() if k in model_df.columns})
    if "model" not in model_df.columns:
        return model_df
    # Merge AUC from model_auc.csv if not already present
    if "auc" not in model_df.columns and auc_table is not None and not auc_table.empty:
        auc_tmp = auc_table.copy()
        if "AUC" in auc_tmp.columns:
            auc_tmp = auc_tmp.rename(columns={"AUC": "auc"})
        if {"model", "auc"}.issubset(auc_tmp.columns):
            model_df = model_df.merge(auc_tmp[["model", "auc"]], on="model", how="left")
    return model_df


def normalize_feature_sensitivity(df):
    """Standardize feature_sensitivity column names from Colab export."""
    if df is None or df.empty:
        return df
    df = df.copy()
    rename_map = {
        "new_avg_outbreak_probability (10% increase)": "new_avg_outbreak_probability",
        "change_in_probability": "delta_probability",
    }
    return df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


# Pre-process
model_comparison    = normalize_model_comparison(model_comparison, auc_df)
feature_sensitivity = normalize_feature_sensitivity(feature_sensitivity)

# Build climate-case correlation on the fly if CSV not present
if climate_case_correlation is None or climate_case_correlation.empty:
    if monthly is not None and "CHSO_cases" in monthly.columns:
        rows = []
        for feat in ["rainfall", "relative_humidity", "temp_mid"]:
            if feat in monthly.columns:
                sub = monthly[[feat, "CHSO_cases"]].apply(pd.to_numeric, errors="coerce").dropna()
                corr = sub.corr().iloc[0, 1] if len(sub) > 1 else np.nan
                rows.append({"feature": feat, "pearson_corr_with_CHSO_cases": corr})
        if rows:
            climate_case_correlation = pd.DataFrame(rows).sort_values("pearson_corr_with_CHSO_cases", ascending=False)


# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Overview",
    "Barangay Analytics",
    "Model Results",
    "Feature Transparency",
    "Forecast & Live Prediction",
])

# ── TAB 1: Overview ──────────────────────────────────────────────────────────
with tab1:
    st.header("Historical Dengue Overview")

    total_months    = len(monthly)
    total_cases     = int(pd.to_numeric(monthly["CHSO_cases"], errors="coerce").fillna(0).sum()) if "CHSO_cases" in monthly.columns else 0
    avg_cases       = pd.to_numeric(monthly["CHSO_cases"], errors="coerce").mean() if "CHSO_cases" in monthly.columns else np.nan
    outbreak_months = int(pd.to_numeric(monthly["outbreak"], errors="coerce").fillna(0).sum()) if "outbreak" in monthly.columns else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Months", total_months)
    col2.metric("Total CHSO Cases", f"{total_cases:,}")
    col3.metric("Average Monthly Cases", safe_metric_value(avg_cases))
    col4.metric("Outbreak Months", outbreak_months)

    st.subheader("Model Prediction Target")
    if meta:
        st.info(
            f"**Problem Definition:** {meta.get('problem_definition', 'Monthly outbreak classification')}  \n"
            f"**Outbreak Definition:** {meta.get('outbreak_definition', '')}"
        )
    else:
        st.info("The model predicts whether a selected month is classified as an outbreak or non-outbreak month.")

    st.subheader("Monthly Dengue Cases")
    if {"Date", "CHSO_cases"}.issubset(monthly.columns):
        if "DOH_cases" in monthly.columns:
            trend_long = monthly[["Date", "CHSO_cases", "DOH_cases"]].melt(id_vars="Date", var_name="Source", value_name="Cases")
            fig_line = px.line(trend_long, x="Date", y="Cases", color="Source", markers=True,
                               title="Monthly Dengue Cases: CHSO and DOH Comparison")
        else:
            fig_line = px.line(monthly, x="Date", y="CHSO_cases", markers=True,
                               title="Monthly Dengue Cases in Baguio City (CHSO)")
        st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Year-Month Heatmap of CHSO Dengue Cases")
    if {"Year", "Month", "CHSO_cases"}.issubset(monthly.columns):
        heat = monthly.pivot_table(index="Year", columns="Month", values="CHSO_cases", aggfunc="sum")
        fig_heat = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
                             title="Year-Month Heatmap of CHSO Dengue Cases")
        fig_heat.update_xaxes(title="Month")
        fig_heat.update_yaxes(title="Year")
        st.plotly_chart(fig_heat, use_container_width=True)

    st.subheader("Rainfall vs Relative Humidity Sized by Dengue Cases")
    if {"rainfall", "relative_humidity", "CHSO_cases"}.issubset(monthly.columns):
        hover_cols = [c for c in ["Date", "temp_mid", "CHSO_cases"] if c in monthly.columns]
        fig_bubble = px.scatter(
            monthly, x="rainfall", y="relative_humidity",
            size="CHSO_cases", color="CHSO_cases", hover_data=hover_cols,
            title="Rainfall vs Relative Humidity Sized by CHSO Dengue Cases",
        )
        st.plotly_chart(fig_bubble, use_container_width=True)

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Climate-Case Correlation")
        if climate_case_correlation is not None and not climate_case_correlation.empty:
            st.dataframe(round_display_columns(climate_case_correlation, ["pearson_corr_with_CHSO_cases"], 4), use_container_width=True)
            if {"feature", "pearson_corr_with_CHSO_cases"}.issubset(climate_case_correlation.columns):
                fig_corr = px.bar(
                    round_display_columns(climate_case_correlation, ["pearson_corr_with_CHSO_cases"], 4),
                    x="feature", y="pearson_corr_with_CHSO_cases", text="pearson_corr_with_CHSO_cases",
                    title="Climate-Case Correlation",
                )
                fig_corr.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("climate_case_correlation.csv is unavailable.")

    with chart_col2:
        st.subheader("Average Monthly Profile")
        if month_profile is not None and not month_profile.empty:
            display_cols = [c for c in ["Month", "MonthName", "CHSO_cases", "rainfall", "relative_humidity", "temp_mid"] if c in month_profile.columns]
            numeric_cols = [c for c in display_cols if c not in ("Month", "MonthName")]
            st.dataframe(round_display_columns(month_profile[display_cols], numeric_cols, 2), use_container_width=True)
            if {"MonthName", "CHSO_cases"}.issubset(month_profile.columns):
                fig_month = px.bar(
                    round_display_columns(month_profile, ["CHSO_cases"], 2),
                    x="MonthName", y="CHSO_cases", text="CHSO_cases",
                    title="Average CHSO Cases by Month",
                )
                fig_month.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.warning("month_profile.csv is unavailable.")

    st.subheader("Climate Profile of Outbreak vs Non-outbreak Months")
    if {"outbreak", "rainfall", "relative_humidity", "temp_mid"}.issubset(monthly.columns):
        climate_profile = monthly.groupby("outbreak", as_index=False)[["rainfall", "relative_humidity", "temp_mid"]].mean()
        climate_profile["Outbreak Status"] = climate_profile["outbreak"].map({0: "Non-outbreak", 1: "Outbreak"})
        climate_long = climate_profile.melt(
            id_vars="Outbreak Status",
            value_vars=["rainfall", "relative_humidity", "temp_mid"],
            var_name="Climate Variable", value_name="Average Value",
        )
        fig_climate = px.bar(
            round_display_columns(climate_long, ["Average Value"], 2),
            x="Climate Variable", y="Average Value", color="Outbreak Status",
            barmode="group", text="Average Value",
            title="Climate Profile of Outbreak vs Non-outbreak Months",
        )
        fig_climate.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_climate, use_container_width=True)

# ── TAB 2: Barangay Analytics ─────────────────────────────────────────────────
with tab2:
    st.header("Barangay Analytics")

    st.subheader("Barangay with the Highest Monthly Dengue Cases")
    if top_barangay_monthly is not None and not top_barangay_monthly.empty:
        st.dataframe(display_clean_barangay_columns(top_barangay_monthly), use_container_width=True)
    else:
        st.warning("top_barangay_monthly.csv is unavailable.")

    st.subheader("Barangays with the Highest Dengue Cases")
    ranking_choice = st.radio("Choose ranking view", ["Three Highest per Year", "Three Highest Overall"], horizontal=True)

    if ranking_choice == "Three Highest per Year":
        if top3_barangays_yearly is not None and not top3_barangays_yearly.empty:
            st.dataframe(display_clean_barangay_columns(top3_barangays_yearly), use_container_width=True)
            if {"Year", "Barangay", "Barangay_cases"}.issubset(top3_barangays_yearly.columns):
                fig_tree = px.treemap(
                    top3_barangays_yearly, path=["Year", "Barangay"], values="Barangay_cases",
                    color="Barangay_cases", title="Three Barangays with the Highest Dengue Cases per Year",
                )
                st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.warning("top3_barangays_yearly.csv is unavailable.")
    else:
        if top3_barangays_overall is not None and not top3_barangays_overall.empty:
            st.dataframe(display_clean_barangay_columns(top3_barangays_overall), use_container_width=True)
            if {"Barangay", "Barangay_cases"}.issubset(top3_barangays_overall.columns):
                fig_overall = px.bar(
                    top3_barangays_overall, x="Barangay", y="Barangay_cases", text="Barangay_cases",
                    title="Three Barangays with the Highest Overall Dengue Cases",
                )
                fig_overall.update_traces(textposition="outside")
                st.plotly_chart(fig_overall, use_container_width=True)
        else:
            st.warning("top3_barangays_overall.csv is unavailable.")

    st.subheader("Barangay Monthly Records")
    if barangay_monthly is not None and not barangay_monthly.empty:
        st.dataframe(display_clean_barangay_columns(barangay_monthly), use_container_width=True)
    else:
        st.warning("barangay_monthly.csv is unavailable.")

# ── TAB 3: Model Results ──────────────────────────────────────────────────────
with tab3:
    st.header("Model Results")

    if meta:
        st.success(f"Selected Model: {meta.get('best_model', 'Unknown')}")

    if model_comparison is not None and not model_comparison.empty:
        display_cols = [c for c in ["model", "f1_score", "precision", "recall", "auc", "brier"] if c in model_comparison.columns]
        st.dataframe(
            round_display_columns(model_comparison[display_cols], [c for c in display_cols if c != "model"], 4),
            use_container_width=True,
        )
        metric_cols = [c for c in ["f1_score", "precision", "recall", "auc"] if c in model_comparison.columns]
        if "model" in model_comparison.columns and metric_cols:
            st.subheader("Model Comparison by Metric")
            results_long = model_comparison.melt(id_vars="model", value_vars=metric_cols, var_name="Metric", value_name="Score")
            fig_model = px.bar(
                round_display_columns(results_long, ["Score"], 4),
                x="model", y="Score", color="Metric", barmode="group", text="Score",
                title="Model Comparison by Metric",
            )
            fig_model.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_model.update_yaxes(range=[0, 1.15])
            st.plotly_chart(fig_model, use_container_width=True)
    else:
        st.warning("model_comparison.csv is unavailable.")

    st.subheader("How to Read the Metrics")
    st.markdown(
        "**Precision** measures how often predicted outbreak months are actually outbreaks. "
        "**Recall** measures how many actual outbreak months the model catches. "
        "**F1 score** balances precision and recall. "
        "**AUC** summarizes how well the model separates outbreak from non-outbreak months across probability thresholds."
    )

    st.subheader("Month-by-Month Test Predictions")
    if test_predictions is not None and not test_predictions.empty:
        total_test   = len(test_predictions)
        correct_test = int(pd.to_numeric(test_predictions["is_correct"], errors="coerce").fillna(0).sum()) if "is_correct" in test_predictions.columns else "N/A"
        c1, c2 = st.columns(2)
        c1.metric("Test Set Months", total_test)
        c2.metric("Correct Predictions", correct_test)
        st.dataframe(test_predictions, use_container_width=True)
    else:
        st.warning("test_predictions.csv is unavailable.")

# ── TAB 4: Feature Transparency ───────────────────────────────────────────────
with tab4:
    st.header("Feature Transparency")

    st.subheader("Primary Contributing Features")
    if feature_importance is not None and not feature_importance.empty:
        st.dataframe(round_display_columns(feature_importance, ["importance_mean", "importance_std"], 6), use_container_width=True)
        if {"feature", "importance_mean"}.issubset(feature_importance.columns):
            fig_importance = px.bar(
                feature_importance.sort_values("importance_mean", ascending=True).tail(15),
                x="importance_mean", y="feature", orientation="h",
                title="Primary Contributing Features",
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("feature_importance.csv is unavailable.")

    st.subheader("Sensitivity Analysis")
    if feature_sensitivity is not None and not feature_sensitivity.empty:
        num_sens_cols = [c for c in ["base_avg_outbreak_probability", "new_avg_outbreak_probability", "delta_probability", "percent_change"] if c in feature_sensitivity.columns]
        st.dataframe(round_display_columns(feature_sensitivity, num_sens_cols, 6), use_container_width=True)
        if {"feature", "delta_probability"}.issubset(feature_sensitivity.columns):
            fig_sens = px.bar(
                round_display_columns(feature_sensitivity, ["delta_probability"], 6),
                x="feature", y="delta_probability", text="delta_probability",
                title="Effect of +10% Change in Climate Variables on Outbreak Probability",
            )
            fig_sens.update_traces(texttemplate="%{text:.6f}", textposition="outside")
            st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.warning("feature_sensitivity.csv is unavailable.")

    st.info(
        "Feature importance and sensitivity analysis explain model behavior. "
        "They do not, by themselves, prove direct biological causation."
    )

# ── TAB 5: Forecast & Live Prediction ────────────────────────────────────────
with tab5:
    st.header("Forecast & Live Prediction")

    st.subheader("Five-Year Forecast")
    if forecast is not None and not forecast.empty:
        st.dataframe(round_display_columns(forecast.head(30), ["predicted_outbreak_probability", "predicted_city_cases_proxy"], 4), use_container_width=True)
        if {"Date", "predicted_outbreak_probability"}.issubset(forecast.columns):
            fig_forecast = px.line(
                forecast,
                x="Date",
                y="predicted_outbreak_probability",
                markers=True,
                title="5-Year Forecasted Outbreak Probability",
            )
            st.plotly_chart(fig_forecast, use_container_width=True)
        if {"Year", "Month", "predicted_outbreak_probability"}.issubset(forecast.columns):
            forecast_heat = forecast.pivot_table(index="Year", columns="Month", values="predicted_outbreak_probability")
            fig_forecast_heat = px.imshow(
                forecast_heat,
                text_auto=".4f",
                aspect="auto",
                color_continuous_scale="Reds",
                title="Forecast Heatmap of Outbreak Probability",
            )
            fig_forecast_heat.update_xaxes(title="Month")
            fig_forecast_heat.update_yaxes(title="Year")
            st.plotly_chart(fig_forecast_heat, use_container_width=True)
    else:
        st.warning("forecast_5yr.csv or forecast_df.csv is unavailable.")

    st.subheader("Three Barangays with the Highest Predicted Risk for Forecast Months")
    if forecast_top3_barangays is not None and not forecast_top3_barangays.empty:
        display_forecast_top3 = round_display_columns(
            forecast_top3_barangays,
            ["overall_share", "recent_share", "seasonal_share", "risk_score_raw", "risk_score", "predicted_outbreak_probability", "predicted_city_cases_proxy", "predicted_barangay_cases_proxy"],
            decimals=4,
        )
        st.dataframe(display_forecast_top3, use_container_width=True)
        if "Date" in forecast_top3_barangays.columns and {"Barangay", "predicted_barangay_cases_proxy"}.issubset(forecast_top3_barangays.columns):
            month_options = forecast_top3_barangays["Date"].dropna().astype(str).unique().tolist()
            if month_options:
                selected_month = st.selectbox("Select forecast month for barangay ranking", month_options)
                selected_barangay_forecast = forecast_top3_barangays[forecast_top3_barangays["Date"].astype(str) == selected_month].copy()
                fig_barangay_forecast = px.bar(
                    round_display_columns(selected_barangay_forecast, ["predicted_barangay_cases_proxy"], decimals=2),
                    x="Barangay",
                    y="predicted_barangay_cases_proxy",
                    color="Barangay",
                    text="predicted_barangay_cases_proxy",
                    title=f"Three Barangays with the Highest Predicted Risk - {selected_month}",
                )
                fig_barangay_forecast.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                st.plotly_chart(fig_barangay_forecast, use_container_width=True)
    else:
        st.warning("forecast_top3_barangays.csv is unavailable.")

    st.markdown("---")
    st.subheader("Live Prediction")
    st.info(
        "Select a target year-month, enter the target month's climate values and the previous three months of dengue cases, then click Predict. The dashboard prepares the lag, rolling, and seasonal model inputs automatically."
    )

    with st.expander("Input guide", expanded=False):
        st.markdown(
            """
**Rainfall, relative humidity, and temperature** should refer to the target month itself. For example, if predicting February 2027, enter the climate values for February 2027. **Cases Last Month**, **Cases 2 Months Ago**, and **Cases 3 Months Ago** should refer to January 2027, December 2026, and November 2026 respectively. The output is a monthly outbreak classification, not a percentage of the population.
"""
        )

    if model is None:
        st.warning("best_model.joblib is unavailable, so live prediction cannot run yet.")
    else:
        feature_columns = meta.get("feature_columns", DEFAULT_FEATURE_COLS) if meta else DEFAULT_FEATURE_COLS

        if forecast is not None and not forecast.empty and "Year" in forecast.columns:
            year_options = sorted(pd.to_numeric(forecast["Year"], errors="coerce").dropna().astype(int).unique().tolist())
        elif "Year" in monthly.columns:
            last_year = int(pd.to_numeric(monthly["Year"], errors="coerce").max())
            year_options = list(range(last_year + 1, last_year + 6))
        else:
            year_options = [2027, 2028, 2029, 2030, 2031]

        month_options = list(range(1, 13))
        select_col1, select_col2 = st.columns(2)
        with select_col1:
            selected_year_num = st.selectbox("Select Year", year_options, index=0)
        with select_col2:
            selected_month_num = st.selectbox(
                "Select Month",
                month_options,
                format_func=lambda x: f"{x} - {month_name_from_number(x)}",
                index=0,
            )

        target_forecast_row = get_forecast_row(forecast, selected_year_num, selected_month_num)

        rainfall_default = get_profile_value(selected_month_num, "rainfall", month_profile, monthly, 0.0)
        humidity_default = get_profile_value(selected_month_num, "relative_humidity", month_profile, monthly, 0.0)
        temp_default     = get_profile_value(selected_month_num, "temp_mid", month_profile, monthly, 0.0)

        if target_forecast_row is not None:
            rainfall_default = float(target_forecast_row.get("rainfall", rainfall_default))
            humidity_default = float(target_forecast_row.get("relative_humidity", humidity_default))
            temp_default     = float(target_forecast_row.get("temp_mid", temp_default))

        rain_min, rain_max = 0.0, max(1000.0, get_reasonable_range(monthly, "rainfall", 0.0, 1000.0)[1])
        rh_min, rh_max     = get_reasonable_range(monthly, "relative_humidity", 60.0, 100.0)
        temp_min, temp_max = 10.0, 35.0
        cases_min, cases_max = get_reasonable_range(monthly, "CHSO_cases", 0.0, 3000.0)

        st.markdown(f"### Target Month: {month_name_from_number(selected_month_num)} {selected_year_num}")
        climate_col1, climate_col2, climate_col3 = st.columns(3)
        with climate_col1:
            rainfall_now = st.slider(
                "Current Rainfall (mm)",
                min_value=float(round(rain_min, 2)),
                max_value=float(round(rain_max, 2)),
                value=float(round(rainfall_default, 2)),
                step=1.0,
            )
        with climate_col2:
            humidity_now = st.slider(
                "Current Relative Humidity (%)",
                min_value=float(round(rh_min, 2)),
                max_value=float(round(rh_max, 2)),
                value=float(round(min(max(humidity_default, rh_min), rh_max), 2)),
                step=0.1,
            )
        with climate_col3:
            temp_now = st.slider(
                "Current Temperature (°C)",
                min_value=float(round(temp_min, 2)),
                max_value=float(round(temp_max, 2)),
                value=float(round(min(max(temp_default, temp_min), temp_max), 2)),
                step=0.1,
            )

        st.markdown("### Recent Dengue Case History")
        if target_forecast_row is not None:
            default_cases_lag_1 = float(target_forecast_row.get("cases_lag_1", 0.0))
            default_cases_lag_2 = float(target_forecast_row.get("cases_lag_2", default_cases_lag_1))
            default_cases_lag_3 = float(target_forecast_row.get("cases_lag_3", default_cases_lag_2))
        else:
            cases_series = pd.to_numeric(monthly["CHSO_cases"], errors="coerce").dropna() if "CHSO_cases" in monthly.columns else pd.Series(dtype=float)
            default_cases_lag_1 = float(cases_series.iloc[-1]) if len(cases_series) >= 1 else 0.0
            default_cases_lag_2 = float(cases_series.iloc[-2]) if len(cases_series) >= 2 else default_cases_lag_1
            default_cases_lag_3 = float(cases_series.iloc[-3]) if len(cases_series) >= 3 else default_cases_lag_2

        max_cases_slider = int(max(cases_max, default_cases_lag_1, default_cases_lag_2, default_cases_lag_3, 1))
        case_col1, case_col2, case_col3 = st.columns(3)
        with case_col1:
            cases_lag_1 = st.slider("Cases Last Month", 0, max_cases_slider, int(round(default_cases_lag_1)), step=1)
        with case_col2:
            cases_lag_2 = st.slider("Cases 2 Months Ago", 0, max_cases_slider, int(round(default_cases_lag_2)), step=1)
        with case_col3:
            cases_lag_3 = st.slider("Cases 3 Months Ago", 0, max_cases_slider, int(round(default_cases_lag_3)), step=1)

        auto_feature_values = build_live_prediction_features(
            year_num=selected_year_num,
            month_num=selected_month_num,
            rainfall_now=rainfall_now,
            humidity_now=humidity_now,
            temp_now=temp_now,
            cases_lag_1=cases_lag_1,
            cases_lag_2=cases_lag_2,
            cases_lag_3=cases_lag_3,
            month_profile_df=month_profile,
            monthly_df=monthly,
            forecast_row=target_forecast_row,
        )

        input_values = {feature: auto_feature_values.get(feature, 0.0) for feature in feature_columns}

        with st.expander("Show automatically prepared model inputs", expanded=False):
            st.dataframe(pd.DataFrame([input_values]), use_container_width=True)

        if st.button("Predict", type="primary"):
            input_df = pd.DataFrame([input_values])
            pred = int(model.predict(input_df)[0])
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(input_df)[0][1])
            else:
                prob = np.nan

            result_col1, result_col2 = st.columns(2)
            result_col1.success(f"Predicted Class: {outbreak_label_from_binary(pred)}")
            result_col2.info(f"Predicted Outbreak Probability: {prob:.4f}" if not pd.isna(prob) else "Probability not available")

            st.caption("0 = Non-outbreak month; 1 = Outbreak month. Probability is the model's estimated likelihood of the outbreak class.")

            st.subheader("Barangays with the Highest Predicted Risk")
            if barangay_risk_profile is not None and not barangay_risk_profile.empty:
                barangay_live = barangay_risk_profile.copy()
                for col in ["overall_share", "recent_share", "seasonal_share"]:
                    if col not in barangay_live.columns:
                        barangay_live[col] = 0.0

                if "Month" in barangay_live.columns:
                    seasonal_subset = barangay_live[pd.to_numeric(barangay_live["Month"], errors="coerce") == selected_month_num].copy()
                    if not seasonal_subset.empty:
                        barangay_live = seasonal_subset

                barangay_live["risk_score_raw"] = (
                    0.50 * pd.to_numeric(barangay_live["seasonal_share"], errors="coerce").fillna(0) +
                    0.30 * pd.to_numeric(barangay_live["recent_share"], errors="coerce").fillna(0) +
                    0.20 * pd.to_numeric(barangay_live["overall_share"], errors="coerce").fillna(0)
                )
                total_score = barangay_live["risk_score_raw"].sum()
                barangay_live["risk_score"] = barangay_live["risk_score_raw"] / total_score if total_score > 0 else 0.0
                city_cases_proxy = float(input_values.get("cases_roll3_mean", 0.0)) * (1 + (0 if pd.isna(prob) else prob))
                barangay_live["predicted_city_cases_proxy"] = city_cases_proxy
                barangay_live["predicted_barangay_cases_proxy"] = barangay_live["risk_score"] * city_cases_proxy
                barangay_live["predicted_barangay_label"] = "Higher Risk"

                keep_cols = [c for c in [
                    "Barangay", "overall_share", "recent_share", "seasonal_share",
                    "risk_score_raw", "risk_score", "predicted_city_cases_proxy",
                    "predicted_barangay_cases_proxy", "predicted_barangay_label",
                ] if c in barangay_live.columns]

                barangay_live_high_risk = barangay_live.sort_values("predicted_barangay_cases_proxy", ascending=False).head(3)
                st.dataframe(
                    round_display_columns(
                        barangay_live_high_risk[keep_cols],
                        ["overall_share", "recent_share", "seasonal_share", "risk_score_raw", "risk_score", "predicted_city_cases_proxy", "predicted_barangay_cases_proxy"],
                        decimals=4,
                    ),
                    use_container_width=True,
                )

                if {"Barangay", "predicted_barangay_cases_proxy"}.issubset(barangay_live_high_risk.columns):
                    fig_live_barangay = px.bar(
                        round_display_columns(barangay_live_high_risk, ["predicted_barangay_cases_proxy"], decimals=2),
                        x="Barangay",
                        y="predicted_barangay_cases_proxy",
                        color="Barangay",
                        text="predicted_barangay_cases_proxy",
                        title=f"Three Barangays with the Highest Predicted Risk for {month_name_from_number(selected_month_num)} {selected_year_num}",
                    )
                    fig_live_barangay.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                    st.plotly_chart(fig_live_barangay, use_container_width=True)

                st.caption("Barangay case values are weighted proxy estimates for prioritization. They are not confirmed case counts.")
            else:
                st.warning("barangay_risk_profile.csv is unavailable.")

st.markdown("---")
st.caption("Baguio City Dengue Outbreak Forecast Dashboard")
