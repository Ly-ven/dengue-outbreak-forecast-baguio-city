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

# Updated to match the current exported dashboard_artifacts ZIP.
ARTIFACT_FILES = {
    "monthly": "monthly_modeling_dataset.csv",
    "model_comparison": "model_comparison.csv",
    "auc_df": "model_auc.csv",
    "feature_importance": "feature_importance.csv",
    "feature_sensitivity": "feature_sensitivity.csv",
    "forecast": "forecast_5yr.csv",
    "barangay_monthly": "barangay_monthly.csv",
    "top_barangay_monthly": "top_barangay_monthly.csv",
    "top3_barangays_yearly": "top3_barangays_yearly.csv",
    "top3_barangays_overall": "top3_barangays_overall.csv",
    "test_predictions": "test_predictions.csv",
    "climate_case_correlation": "climate_case_correlation.csv",
    "month_profile": "month_profile.csv",
    "forecast_top3_barangays": "forecast_top3_barangays.csv",
    "forecast_barangay_ranking": "forecast_barangay_ranking.csv",
    "barangay_risk_profile": "barangay_risk_profile.csv",
}

CORE_FILES = [
    "monthly_modeling_dataset.csv",
    "forecast_5yr.csv",
    "model_comparison.csv",
    "best_model.joblib",
    "meta.json",
    "feature_columns.json",
]

st.markdown(
    """
<style>
.block-container {padding-top: 1.4rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
.small-caption {font-size: 0.85rem; color: #6c757d;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Baguio City Dengue Forecast Dashboard")
st.caption("Interactive dashboard for dengue outbreak forecasting using localized climate and epidemiological data.")


# =========================
# LOADING HELPERS
# =========================
def artifact_path(filename: str) -> Path:
    return ARTIFACTS_DIR / filename


def load_csv(filename: str):
    path = artifact_path(filename)
    if path.exists():
        return pd.read_csv(path)
    return None


def load_json(filename: str):
    path = artifact_path(filename)
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def parse_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in ["Year", "Month"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


@st.cache_data(show_spinner=False)
def load_all_artifacts():
    loaded = {}
    for key, filename in ARTIFACT_FILES.items():
        df = load_csv(filename)
        loaded[key] = parse_dates(df) if df is not None else None
    loaded["meta"] = load_json("meta.json")
    loaded["feature_columns"] = load_json("feature_columns.json")
    return loaded


@st.cache_resource(show_spinner=False)
def load_model():
    path = artifact_path("best_model.joblib")
    if path.exists():
        return joblib.load(path)
    return None


artifacts = load_all_artifacts()
model = load_model()
meta = artifacts.get("meta") or {}

feature_cols_from_meta = (
    artifacts.get("feature_columns")
    or meta.get("feature_columns")
    or DEFAULT_FEATURE_COLS
)

if not isinstance(feature_cols_from_meta, list):
    feature_cols_from_meta = DEFAULT_FEATURE_COLS

# Extract dataframes
monthly = artifacts.get("monthly")
model_comparison = artifacts.get("model_comparison")
auc_df = artifacts.get("auc_df")
feature_importance = artifacts.get("feature_importance")
feature_sensitivity = artifacts.get("feature_sensitivity")
forecast = artifacts.get("forecast")
barangay_monthly = artifacts.get("barangay_monthly")
top_barangay_monthly = artifacts.get("top_barangay_monthly")
top3_barangays_yearly = artifacts.get("top3_barangays_yearly")
top3_barangays_overall = artifacts.get("top3_barangays_overall")
test_predictions = artifacts.get("test_predictions")
climate_case_correlation = artifacts.get("climate_case_correlation")
month_profile = artifacts.get("month_profile")
forecast_top3_barangays = artifacts.get("forecast_top3_barangays")
forecast_barangay_ranking = artifacts.get("forecast_barangay_ranking")
barangay_risk_profile = artifacts.get("barangay_risk_profile")


# =========================
# GENERAL HELPERS
# =========================
def month_name_from_number(month_num):
    names = {
        1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
        7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
    }
    try:
        return names.get(int(month_num), str(month_num))
    except Exception:
        return str(month_num)


def month_label(year, month):
    try:
        return f"{month_name_from_number(month)} {int(year)}"
    except Exception:
        return "N/A"


def safe_metric_value(value, decimals=2, suffix=""):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}{suffix}"
    except Exception:
        return "N/A"


def safe_int(value, default=0):
    try:
        if pd.isna(value):
            return default
        return int(round(float(value)))
    except Exception:
        return default


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def round_display_columns(df, columns, decimals=2):
    if df is None:
        return None
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
    return df


# Backward-compatible alias for older code blocks.
def round_df(df, columns, decimals=2):
    return round_display_columns(df, columns, decimals)


def available_columns(df, preferred_cols):
    if df is None:
        return []
    return [col for col in preferred_cols if col in df.columns]


def make_month_order(df):
    df = df.copy()
    if "Month" in df.columns:
        df["MonthName"] = df["Month"].apply(month_name_from_number)
        df["MonthOrder"] = pd.to_numeric(df["Month"], errors="coerce")
    return df


def missing_files():
    missing = []
    for filename in CORE_FILES:
        if not artifact_path(filename).exists():
            missing.append(filename)
    return missing


def optional_missing_files():
    missing = []
    for filename in ARTIFACT_FILES.values():
        if not artifact_path(filename).exists() and filename not in CORE_FILES:
            missing.append(filename)
    return sorted(set(missing))


def get_first_existing_row(df, year, month):
    if df is None or df.empty or not {"Year", "Month"}.issubset(df.columns):
        return None
    mask = (pd.to_numeric(df["Year"], errors="coerce") == int(year)) & (pd.to_numeric(df["Month"], errors="coerce") == int(month))
    if mask.any():
        return df.loc[mask].iloc[0]
    return None


def get_month_profile_value(month, col, default):
    if month_profile is None or month_profile.empty or "Month" not in month_profile.columns or col not in month_profile.columns:
        return default
    month_data = month_profile[pd.to_numeric(month_profile["Month"], errors="coerce") == int(month)]
    if month_data.empty:
        return default
    return safe_float(month_data[col].iloc[0], default)


def build_slider_bounds(col, default_value, sources, fallback_min, fallback_max, lower_floor=None, pad_ratio=0.08):
    values = []
    for df in sources:
        if df is not None and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            if not s.empty:
                values.extend(s.tolist())
    values.extend([fallback_min, fallback_max, default_value])

    clean_values = [safe_float(v, np.nan) for v in values]
    clean_values = [v for v in clean_values if not pd.isna(v)]
    if not clean_values:
        clean_values = [fallback_min, fallback_max, default_value]

    min_value = min(clean_values)
    max_value = max(clean_values)
    span = max(max_value - min_value, 1.0)
    min_value -= span * pad_ratio
    max_value += span * pad_ratio

    if lower_floor is not None:
        min_value = max(float(lower_floor), min_value)

    min_value = float(np.floor(min_value * 10) / 10)
    max_value = float(np.ceil(max_value * 10) / 10)
    default_value = float(np.clip(default_value, min_value, max_value))

    if min_value >= max_value:
        max_value = min_value + 1.0

    return min_value, max_value, default_value


def get_lag_value_from_forecast_or_profile(year_month_pairs, idx, col, default):
    try:
        y, m = year_month_pairs[idx]
    except Exception:
        return default

    row = get_first_existing_row(forecast, y, m)
    if row is not None and col in row.index:
        return safe_float(row.get(col), default)

    return get_month_profile_value(m, col, default)


def get_outbreak_probability(model_obj, input_df):
    if not hasattr(model_obj, "predict_proba"):
        return np.nan

    proba = model_obj.predict_proba(input_df)
    if proba is None or len(proba) == 0:
        return np.nan

    classes = getattr(model_obj, "classes_", None)
    if classes is not None and 1 in list(classes):
        class_index = list(classes).index(1)
    else:
        class_index = 1 if proba.shape[1] > 1 else 0

    return float(proba[0][class_index])


def dataframe_with_preferred_columns(df, preferred_cols, round_cols=None, decimals=4):
    if df is None:
        return None
    cols = available_columns(df, preferred_cols)
    display = df[cols].copy() if cols else df.copy()
    return round_display_columns(display, round_cols or [], decimals)


# =========================
# SIDEBAR STATUS
# =========================
st.sidebar.header("Dashboard Status")
core_missing = missing_files()

if monthly is None or core_missing:
    st.sidebar.error("Required artifacts are missing.")
    st.error("The dashboard cannot start because one or more required files are missing from the `artifacts` folder.")
    st.markdown("**Required files:**")
    for filename in CORE_FILES:
        icon = "✅" if artifact_path(filename).exists() else "❌"
        st.code(f"{icon} artifacts/{filename}")
    st.stop()

st.sidebar.success("Artifacts loaded successfully")

if meta:
    st.sidebar.info(f"Best Model: {meta.get('best_model', 'Unknown')}")
    threshold_val = meta.get("outbreak_threshold_cases")
    if isinstance(threshold_val, (int, float)):
        st.sidebar.info(f"Outbreak Threshold: {threshold_val:.2f} cases")
    if meta.get("forecast_period"):
        st.sidebar.info(f"Forecast Period: {meta.get('forecast_period')}")

if model is None:
    st.sidebar.warning("Model file was not loaded. Live prediction is disabled.")

missing_optional = optional_missing_files()
if missing_optional:
    with st.sidebar.expander("Optional files not found"):
        for filename in missing_optional:
            st.code(f"artifacts/{filename}")


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


# =========================
# TAB 1: OVERVIEW
# =========================
with tab1:
    st.header("Historical Dengue Overview")

    if monthly is not None and not monthly.empty:
        total_months = len(monthly)
        total_cases = safe_int(pd.to_numeric(monthly.get("CHSO_cases"), errors="coerce").fillna(0).sum()) if "CHSO_cases" in monthly.columns else 0
        avg_cases = pd.to_numeric(monthly.get("CHSO_cases"), errors="coerce").mean() if "CHSO_cases" in monthly.columns else np.nan
        outbreak_months = safe_int(pd.to_numeric(monthly.get("outbreak"), errors="coerce").fillna(0).sum()) if "outbreak" in monthly.columns else 0
        threshold_val = meta.get("outbreak_threshold_cases", np.nan)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Months", f"{total_months:,}")
        col2.metric("Total CHSO Cases", f"{total_cases:,}")
        col3.metric("Average Monthly Cases", safe_metric_value(avg_cases, 2))
        col4.metric("Outbreak Months", f"{outbreak_months:,}")

        if isinstance(threshold_val, (int, float)):
            st.caption(f"Outbreak threshold used by the artifacts: {threshold_val:.2f} monthly CHSO cases.")
        elif meta.get("outbreak_definition"):
            st.caption(meta.get("outbreak_definition"))

        st.subheader("Monthly Dengue Cases Trend")
        if {"Date", "CHSO_cases"}.issubset(monthly.columns):
            if "DOH_cases" in monthly.columns:
                trend_long = monthly[["Date", "CHSO_cases", "DOH_cases"]].melt(
                    id_vars="Date",
                    var_name="Source",
                    value_name="Cases",
                )
                fig_line = px.line(
                    trend_long,
                    x="Date",
                    y="Cases",
                    color="Source",
                    markers=True,
                    title="CHSO vs DOH Monthly Dengue Counts",
                )
            else:
                fig_line = px.line(
                    monthly,
                    x="Date",
                    y="CHSO_cases",
                    markers=True,
                    title="Monthly Dengue Cases in Baguio City (CHSO)",
                )
            fig_line.update_layout(hovermode="x unified")
            st.plotly_chart(fig_line, use_container_width=True)

        st.subheader("Seasonal Patterns")
        col_left, col_right = st.columns(2)

        with col_left:
            if {"Year", "Month", "CHSO_cases"}.issubset(monthly.columns):
                heat = monthly.pivot_table(
                    index="Year",
                    columns="Month",
                    values="CHSO_cases",
                    aggfunc="sum",
                )
                heat = heat.sort_index(axis=0).sort_index(axis=1)
                heat.columns = [month_name_from_number(m) for m in heat.columns]
                fig_heat = px.imshow(
                    heat,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale="YlOrRd",
                    title="Year-Month Heatmap of Dengue Cases",
                )
                fig_heat.update_xaxes(title="Month")
                fig_heat.update_yaxes(title="Year")
                st.plotly_chart(fig_heat, use_container_width=True)

        with col_right:
            if climate_case_correlation is not None and not climate_case_correlation.empty:
                corr_display = climate_case_correlation.copy()
                if "pearson_corr_with_CHSO_cases" in corr_display.columns:
                    corr_display["pearson_corr_with_CHSO_cases"] = pd.to_numeric(
                        corr_display["pearson_corr_with_CHSO_cases"], errors="coerce"
                    )
                    fig_corr = px.bar(
                        corr_display,
                        x="feature",
                        y="pearson_corr_with_CHSO_cases",
                        text="pearson_corr_with_CHSO_cases",
                        title="Climate-Case Correlation",
                        labels={
                            "feature": "Climate Variable",
                            "pearson_corr_with_CHSO_cases": "Pearson Correlation",
                        },
                    )
                    fig_corr.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                    st.plotly_chart(fig_corr, use_container_width=True)

        if month_profile is not None and not month_profile.empty:
            st.subheader("Average Monthly Cases by Month")
            profile_display = make_month_order(month_profile)
            if {"MonthName", "CHSO_cases"}.issubset(profile_display.columns):
                fig_month = px.bar(
                    profile_display.sort_values("MonthOrder") if "MonthOrder" in profile_display.columns else profile_display,
                    x="MonthName",
                    y="CHSO_cases",
                    text="CHSO_cases",
                    title="Average CHSO Cases by Month",
                    labels={"MonthName": "Month", "CHSO_cases": "Average Cases"},
                )
                fig_month.update_traces(texttemplate="%{text:.1f}", textposition="outside")
                st.plotly_chart(fig_month, use_container_width=True)

            with st.expander("View monthly profile data"):
                display_cols = ["Month", "MonthName", "CHSO_cases", "rainfall", "relative_humidity", "temp_mid"]
                display = dataframe_with_preferred_columns(
                    month_profile,
                    display_cols,
                    ["CHSO_cases", "rainfall", "relative_humidity", "temp_mid"],
                    2,
                )
                st.dataframe(display, use_container_width=True)

        if {"outbreak", "rainfall", "relative_humidity", "temp_mid"}.issubset(monthly.columns):
            st.subheader("Climate Profile: Outbreak vs Non-outbreak Months")
            climate_profile = monthly.groupby("outbreak", as_index=False)[["rainfall", "relative_humidity", "temp_mid"]].mean()
            climate_profile["Outbreak Status"] = climate_profile["outbreak"].map({0: "Non-outbreak", 1: "Outbreak"}).fillna("Unknown")
            climate_long = climate_profile.melt(
                id_vars="Outbreak Status",
                value_vars=["rainfall", "relative_humidity", "temp_mid"],
                var_name="Climate Variable",
                value_name="Average Value",
            )
            fig_climate = px.bar(
                climate_long,
                x="Climate Variable",
                y="Average Value",
                color="Outbreak Status",
                barmode="group",
                text="Average Value",
                title="Climate Variable Averages by Outbreak Status",
            )
            fig_climate.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(fig_climate, use_container_width=True)


# =========================
# TAB 2: BARANGAY ANALYTICS
# =========================
with tab2:
    st.header("Barangay Analytics")

    col_a, col_b, col_c = st.columns(3)

    if barangay_monthly is not None and not barangay_monthly.empty:
        total_barangays = barangay_monthly["Barangay"].nunique() if "Barangay" in barangay_monthly.columns else 0
        total_barangay_cases = safe_int(pd.to_numeric(barangay_monthly.get("Barangay_cases"), errors="coerce").fillna(0).sum()) if "Barangay_cases" in barangay_monthly.columns else 0
        col_a.metric("Barangays in Records", f"{total_barangays:,}")
        col_b.metric("Barangay Case Records", f"{len(barangay_monthly):,}")
        col_c.metric("Total Barangay Cases", f"{total_barangay_cases:,}")

    if top_barangay_monthly is not None and not top_barangay_monthly.empty:
        st.subheader("Barangay with Highest Monthly Dengue Cases")
        display = dataframe_with_preferred_columns(
            top_barangay_monthly,
            ["Date", "Year", "Month", "Top_Barangay", "Top_Barangay_Cases"],
            ["Top_Barangay_Cases"],
            0,
        )
        st.dataframe(display, use_container_width=True)

        if {"Date", "Top_Barangay_Cases", "Top_Barangay"}.issubset(top_barangay_monthly.columns):
            fig_top_monthly = px.bar(
                top_barangay_monthly,
                x="Date",
                y="Top_Barangay_Cases",
                color="Top_Barangay",
                title="Highest Barangay Case Count per Month",
                labels={"Top_Barangay_Cases": "Cases", "Top_Barangay": "Barangay"},
            )
            st.plotly_chart(fig_top_monthly, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        if top3_barangays_yearly is not None and not top3_barangays_yearly.empty:
            st.subheader("Top 3 Barangays per Year")
            display = dataframe_with_preferred_columns(
                top3_barangays_yearly,
                ["Year", "Barangay", "Barangay_cases", "rank_within_year"],
                ["Barangay_cases", "rank_within_year"],
                0,
            )
            st.dataframe(display, use_container_width=True)

            if {"Year", "Barangay", "Barangay_cases"}.issubset(top3_barangays_yearly.columns):
                fig_yearly = px.bar(
                    top3_barangays_yearly,
                    x="Year",
                    y="Barangay_cases",
                    color="Barangay",
                    barmode="group",
                    title="Three Barangays with the Highest Cases per Year",
                    labels={"Barangay_cases": "Cases"},
                )
                st.plotly_chart(fig_yearly, use_container_width=True)

    with col_right:
        if top3_barangays_overall is not None and not top3_barangays_overall.empty:
            st.subheader("Top 3 Barangays Overall")
            display = dataframe_with_preferred_columns(
                top3_barangays_overall,
                ["rank", "Barangay", "Barangay_cases"],
                ["Barangay_cases", "rank"],
                0,
            )
            st.dataframe(display, use_container_width=True)

            if {"Barangay", "Barangay_cases"}.issubset(top3_barangays_overall.columns):
                fig_overall = px.bar(
                    top3_barangays_overall.sort_values("Barangay_cases", ascending=True),
                    x="Barangay_cases",
                    y="Barangay",
                    orientation="h",
                    text="Barangay_cases",
                    title="Overall Highest Barangay Case Counts",
                    labels={"Barangay_cases": "Cases"},
                )
                fig_overall.update_traces(texttemplate="%{text:.0f}", textposition="outside")
                st.plotly_chart(fig_overall, use_container_width=True)

    if barangay_monthly is not None and not barangay_monthly.empty and {"Barangay", "Date", "Barangay_cases"}.issubset(barangay_monthly.columns):
        st.subheader("Barangay Monthly Records")
        barangay_options = sorted(barangay_monthly["Barangay"].dropna().astype(str).unique())
        selected_barangay = st.selectbox("Select Barangay", barangay_options, index=0)
        barangay_selected = barangay_monthly[barangay_monthly["Barangay"].astype(str) == selected_barangay].copy()

        fig_barangay = px.line(
            barangay_selected,
            x="Date",
            y="Barangay_cases",
            markers=True,
            title=f"Monthly Dengue Cases: {selected_barangay}",
            labels={"Barangay_cases": "Cases"},
        )
        fig_barangay.update_layout(hovermode="x unified")
        st.plotly_chart(fig_barangay, use_container_width=True)

        with st.expander("View full barangay monthly table"):
            st.dataframe(
                round_display_columns(barangay_monthly, ["Barangay_cases"], 0),
                use_container_width=True,
            )


# =========================
# TAB 3: MODEL RESULTS
# =========================
with tab3:
    st.header("Model Results")

    if meta:
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Model", meta.get("best_model", "Unknown"))
        col2.metric("Training Months", meta.get("train_months", "N/A"))
        col3.metric("Testing Months", meta.get("test_months", "N/A"))
        if meta.get("test_split_note"):
            st.caption(meta.get("test_split_note"))

    if model_comparison is not None and not model_comparison.empty:
        st.subheader("Model Performance Comparison")
        display_cols = [
            "Model", "Accuracy", "F1 Score", "Precision", "Recall",
            "AUC (Supplementary)", "Reliability (Brier)",
        ]
        display = dataframe_with_preferred_columns(
            model_comparison,
            display_cols,
            ["Accuracy", "F1 Score", "Precision", "Recall", "AUC (Supplementary)", "Reliability (Brier)"],
            4,
        )
        st.dataframe(display, use_container_width=True)

        metric_cols = [c for c in ["Accuracy", "F1 Score", "Precision", "Recall", "AUC (Supplementary)"] if c in model_comparison.columns]
        if "Model" in model_comparison.columns and metric_cols:
            metric_long = model_comparison[["Model"] + metric_cols].melt(
                id_vars="Model",
                var_name="Metric",
                value_name="Score",
            )
            fig_metrics = px.bar(
                metric_long,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                text="Score",
                title="Classification Performance by Model",
            )
            fig_metrics.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_metrics.update_yaxes(range=[0, 1.05])
            st.plotly_chart(fig_metrics, use_container_width=True)

        if "Model" in model_comparison.columns and "Accuracy" in model_comparison.columns:
            best_row = model_comparison.loc[pd.to_numeric(model_comparison["Accuracy"], errors="coerce").idxmax()]
            st.metric("Highest Accuracy", f"{safe_float(best_row['Accuracy']):.4f}", delta=f"Model: {best_row['Model']}")

    if auc_df is not None and not auc_df.empty:
        with st.expander("View supplementary AUC table"):
            st.dataframe(round_display_columns(auc_df, ["AUC"], 4), use_container_width=True)

    if test_predictions is not None and not test_predictions.empty:
        st.subheader("Month-by-Month Test Predictions")
        correct = safe_int(pd.to_numeric(test_predictions.get("is_correct"), errors="coerce").fillna(0).sum()) if "is_correct" in test_predictions.columns else 0
        accuracy = correct / len(test_predictions) if len(test_predictions) > 0 else 0
        st.metric("Test Set Accuracy", f"{accuracy:.4f}", delta=f"{correct} / {len(test_predictions)} correct")

        display = dataframe_with_preferred_columns(
            test_predictions,
            ["Date", "Year", "Month", "CHSO_cases", "outbreak", "predicted_outbreak", "predicted_probability", "actual_label", "predicted_label", "is_correct"],
            ["CHSO_cases", "predicted_probability"],
            4,
        )
        st.dataframe(display, use_container_width=True)

        if {"Date", "predicted_probability"}.issubset(test_predictions.columns):
            fig_test = px.line(
                test_predictions,
                x="Date",
                y="predicted_probability",
                markers=True,
                title="Predicted Outbreak Probability on Test Months",
                labels={"predicted_probability": "Predicted Probability"},
            )
            fig_test.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_test, use_container_width=True)


# =========================
# TAB 4: FEATURE TRANSPARENCY
# =========================
with tab4:
    st.header("Feature Transparency")

    if feature_importance is not None and not feature_importance.empty:
        st.subheader("Feature Importance")
        imp_display = feature_importance.copy()
        if "importance_mean" in imp_display.columns:
            imp_display["importance_mean"] = pd.to_numeric(imp_display["importance_mean"], errors="coerce")
            imp_display = imp_display.sort_values("importance_mean", ascending=False)

        st.dataframe(
            round_display_columns(imp_display, ["importance_mean", "importance_std"], 6),
            use_container_width=True,
        )

        if {"feature", "importance_mean"}.issubset(imp_display.columns):
            top_imp = imp_display.head(10).sort_values("importance_mean", ascending=True)
            fig_imp = px.bar(
                top_imp,
                x="importance_mean",
                y="feature",
                orientation="h",
                text="importance_mean",
                title="Top 10 Most Important Features",
                labels={"importance_mean": "Importance Mean", "feature": "Feature"},
            )
            fig_imp.update_traces(texttemplate="%{text:.6f}", textposition="outside")
            st.plotly_chart(fig_imp, use_container_width=True)

    if feature_sensitivity is not None and not feature_sensitivity.empty:
        st.subheader("Sensitivity Analysis (+10% Climate Change)")
        st.dataframe(
            round_display_columns(
                feature_sensitivity,
                ["base_avg_outbreak_probability", "new_avg_outbreak_probability (10% increase)", "change_in_probability", "percent_change"],
                6,
            ),
            use_container_width=True,
        )

        if {"feature", "change_in_probability"}.issubset(feature_sensitivity.columns):
            fig_sens = px.bar(
                feature_sensitivity,
                x="feature",
                y="change_in_probability",
                text="change_in_probability",
                title="Change in Average Outbreak Probability After +10% Feature Increase",
                labels={"change_in_probability": "Change in Probability", "feature": "Feature"},
            )
            fig_sens.update_traces(texttemplate="%{text:.6f}", textposition="outside")
            st.plotly_chart(fig_sens, use_container_width=True)

    if barangay_risk_profile is not None and not barangay_risk_profile.empty:
        with st.expander("View barangay risk profile used for forecast ranking"):
            st.dataframe(
                round_display_columns(barangay_risk_profile, ["seasonal_share", "overall_share", "recent_share"], 6),
                use_container_width=True,
            )


# =========================
# TAB 5: FORECAST & LIVE PREDICTION
# =========================
with tab5:
    st.header("Forecast & Live Prediction")

    if forecast is not None and not forecast.empty:
        st.subheader("Five-Year Forecast")

        forecast_metrics = st.columns(4)
        forecast_metrics[0].metric("Forecast Months", f"{len(forecast):,}")
        if "predicted_outbreak" in forecast.columns:
            forecast_metrics[1].metric("Predicted Outbreak Months", f"{safe_int(pd.to_numeric(forecast['predicted_outbreak'], errors='coerce').fillna(0).sum()):,}")
        if "predicted_outbreak_probability" in forecast.columns:
            avg_prob = pd.to_numeric(forecast["predicted_outbreak_probability"], errors="coerce").mean()
            forecast_metrics[2].metric("Average Probability", safe_metric_value(avg_prob, 4))
            peak_idx = pd.to_numeric(forecast["predicted_outbreak_probability"], errors="coerce").idxmax()
            peak_row = forecast.loc[peak_idx]
            forecast_metrics[3].metric("Peak Month", month_label(peak_row.get("Year"), peak_row.get("Month")))

        preferred_forecast_cols = [
            "Date", "Year", "Month", "rainfall", "relative_humidity", "temp_mid",
            "predicted_outbreak_probability", "predicted_outbreak_probability_lower",
            "predicted_outbreak_probability_upper", "predicted_outbreak",
            "predicted_label", "predicted_city_cases_proxy",
        ]
        forecast_display = dataframe_with_preferred_columns(
            forecast.head(30),
            preferred_forecast_cols,
            [
                "rainfall", "relative_humidity", "temp_mid",
                "predicted_outbreak_probability", "predicted_outbreak_probability_lower",
                "predicted_outbreak_probability_upper", "predicted_city_cases_proxy",
            ],
            4,
        )
        st.dataframe(forecast_display, use_container_width=True)

        if {"Date", "predicted_outbreak_probability"}.issubset(forecast.columns):
            fig_forecast = px.line(
                forecast,
                x="Date",
                y="predicted_outbreak_probability",
                markers=True,
                title="5-Year Forecasted Outbreak Probability",
                labels={"predicted_outbreak_probability": "Outbreak Probability"},
            )
            fig_forecast.update_layout(hovermode="x unified")
            fig_forecast.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_forecast, use_container_width=True)

        if {"Year", "Month", "predicted_outbreak_probability"}.issubset(forecast.columns):
            forecast_heat = forecast.pivot_table(
                index="Year",
                columns="Month",
                values="predicted_outbreak_probability",
                aggfunc="mean",
            )
            forecast_heat = forecast_heat.sort_index(axis=0).sort_index(axis=1)
            forecast_heat.columns = [month_name_from_number(m) for m in forecast_heat.columns]
            fig_forecast_heat = px.imshow(
                forecast_heat,
                text_auto=".4f",
                aspect="auto",
                color_continuous_scale="Reds",
                title="Forecast Heatmap - Outbreak Probability",
            )
            fig_forecast_heat.update_xaxes(title="Month")
            fig_forecast_heat.update_yaxes(title="Year")
            st.plotly_chart(fig_forecast_heat, use_container_width=True)

    if forecast_top3_barangays is not None and not forecast_top3_barangays.empty:
        st.subheader("Forecasted Top 3 Barangays")

        year_filter_options = sorted(pd.to_numeric(forecast_top3_barangays.get("Year"), errors="coerce").dropna().astype(int).unique()) if "Year" in forecast_top3_barangays.columns else []
        if year_filter_options:
            selected_forecast_year = st.selectbox("Select forecast year for barangay ranking", year_filter_options, index=0)
            top3_filtered = forecast_top3_barangays[pd.to_numeric(forecast_top3_barangays["Year"], errors="coerce") == selected_forecast_year].copy()
        else:
            selected_forecast_year = None
            top3_filtered = forecast_top3_barangays.copy()

        display = dataframe_with_preferred_columns(
            top3_filtered,
            [
                "Date", "Year", "Month", "Barangay", "risk_score", "predicted_outbreak_probability",
                "predicted_city_cases_proxy", "predicted_barangay_cases_proxy", "predicted_barangay_label",
            ],
            ["risk_score", "predicted_outbreak_probability", "predicted_city_cases_proxy", "predicted_barangay_cases_proxy"],
            4,
        )
        st.dataframe(display, use_container_width=True)

        if {"Date", "Barangay", "predicted_barangay_cases_proxy"}.issubset(top3_filtered.columns):
            fig_top3_forecast = px.bar(
                top3_filtered,
                x="Date",
                y="predicted_barangay_cases_proxy",
                color="Barangay",
                barmode="group",
                title="Top 3 Forecasted Barangays by Month",
                labels={"predicted_barangay_cases_proxy": "Predicted Barangay Cases Proxy"},
            )
            st.plotly_chart(fig_top3_forecast, use_container_width=True)

    if forecast_barangay_ranking is not None and not forecast_barangay_ranking.empty:
        with st.expander("View full forecast barangay ranking"):
            st.dataframe(
                round_display_columns(
                    forecast_barangay_ranking,
                    ["overall_share", "recent_share", "seasonal_share", "risk_score_raw", "risk_score", "predicted_outbreak_probability", "predicted_city_cases_proxy", "predicted_barangay_cases_proxy"],
                    6,
                ),
                use_container_width=True,
            )

    # Live prediction
    st.markdown("---")
    st.subheader("Live Prediction")

    if model is None:
        st.warning("Model not loaded. Cannot run live prediction.")
    else:
        if forecast is not None and "Year" in forecast.columns:
            year_options = sorted(pd.to_numeric(forecast["Year"], errors="coerce").dropna().astype(int).unique())
        else:
            year_options = list(range(2027, 2032))

        col1, col2 = st.columns(2)
        with col1:
            selected_year = st.selectbox("Year", year_options, index=0)
        with col2:
            selected_month = st.selectbox(
                "Month",
                list(range(1, 13)),
                format_func=lambda x: f"{x} - {month_name_from_number(x)}",
                index=0,
            )

        default_rain = get_month_profile_value(selected_month, "rainfall", 100.0)
        default_humidity = get_month_profile_value(selected_month, "relative_humidity", 80.0)
        default_temp = get_month_profile_value(selected_month, "temp_mid", 25.0)

        forecast_row = get_first_existing_row(forecast, selected_year, selected_month)
        if forecast_row is not None:
            default_rain = safe_float(forecast_row.get("rainfall"), default_rain)
            default_humidity = safe_float(forecast_row.get("relative_humidity"), default_humidity)
            default_temp = safe_float(forecast_row.get("temp_mid"), default_temp)
            st.caption("Default values are taken from the exported forecast row for the selected month.")
        else:
            st.caption("Default values are taken from historical monthly averages.")

        st.markdown(f"**Target Month: {month_name_from_number(selected_month)} {selected_year}**")

        rain_min, rain_max, default_rain = build_slider_bounds(
            "rainfall", default_rain, [forecast, month_profile, monthly], fallback_min=0.0, fallback_max=1500.0, lower_floor=0.0
        )
        hum_min, hum_max, default_humidity = build_slider_bounds(
            "relative_humidity", default_humidity, [forecast, month_profile, monthly], fallback_min=60.0, fallback_max=100.0, lower_floor=0.0
        )
        temp_min, temp_max, default_temp = build_slider_bounds(
            "temp_mid", default_temp, [forecast, month_profile, monthly], fallback_min=10.0, fallback_max=35.0
        )

        c1, c2, c3 = st.columns(3)
        with c1:
            rainfall = st.slider("Rainfall", rain_min, rain_max, float(default_rain), step=1.0)
        with c2:
            humidity = st.slider("Relative Humidity", hum_min, hum_max, float(default_humidity), step=0.5)
        with c3:
            temp = st.slider("Temperature / temp_mid", temp_min, temp_max, float(default_temp), step=0.5)

        st.markdown("**Recent Dengue Cases**")
        if monthly is not None and "CHSO_cases" in monthly.columns:
            cases_series = pd.to_numeric(monthly["CHSO_cases"], errors="coerce").dropna()
            default_lag1 = safe_float(cases_series.iloc[-1], 50.0) if len(cases_series) >= 1 else 50.0
            default_lag2 = safe_float(cases_series.iloc[-2], default_lag1) if len(cases_series) >= 2 else default_lag1
            default_lag3 = safe_float(cases_series.iloc[-3], default_lag2) if len(cases_series) >= 3 else default_lag2
        else:
            default_lag1 = default_lag2 = default_lag3 = 50.0

        if forecast_row is not None:
            default_lag1 = safe_float(forecast_row.get("cases_lag_1"), default_lag1)
            default_lag2 = safe_float(forecast_row.get("cases_lag_2"), default_lag2)
            default_lag3 = safe_float(forecast_row.get("cases_lag_3"), default_lag3)

        d1, d2, d3 = st.columns(3)
        with d1:
            cases_lag1 = st.number_input("Cases - Last Month", min_value=0, value=safe_int(default_lag1, 0), step=10)
        with d2:
            cases_lag2 = st.number_input("Cases - 2 Months Ago", min_value=0, value=safe_int(default_lag2, 0), step=10)
        with d3:
            cases_lag3 = st.number_input("Cases - 3 Months Ago", min_value=0, value=safe_int(default_lag3, 0), step=10)

        lag_periods = []
        for lag in range(1, 4):
            y = int(selected_year)
            m = int(selected_month) - lag
            while m <= 0:
                y -= 1
                m += 12
            lag_periods.append((y, m))

        features = {
            "rainfall": rainfall,
            "relative_humidity": humidity,
            "temp_mid": temp,
            "cases_lag_1": cases_lag1,
            "cases_lag_2": cases_lag2,
            "cases_lag_3": cases_lag3,
            "rainfall_lag_1": get_lag_value_from_forecast_or_profile(lag_periods, 0, "rainfall", default_rain),
            "rainfall_lag_2": get_lag_value_from_forecast_or_profile(lag_periods, 1, "rainfall", default_rain),
            "rainfall_lag_3": get_lag_value_from_forecast_or_profile(lag_periods, 2, "rainfall", default_rain),
            "relative_humidity_lag_1": get_lag_value_from_forecast_or_profile(lag_periods, 0, "relative_humidity", default_humidity),
            "relative_humidity_lag_2": get_lag_value_from_forecast_or_profile(lag_periods, 1, "relative_humidity", default_humidity),
            "relative_humidity_lag_3": get_lag_value_from_forecast_or_profile(lag_periods, 2, "relative_humidity", default_humidity),
            "temp_mid_lag_1": get_lag_value_from_forecast_or_profile(lag_periods, 0, "temp_mid", default_temp),
            "temp_mid_lag_2": get_lag_value_from_forecast_or_profile(lag_periods, 1, "temp_mid", default_temp),
            "temp_mid_lag_3": get_lag_value_from_forecast_or_profile(lag_periods, 2, "temp_mid", default_temp),
            "cases_roll3_mean": float(np.mean([cases_lag1, cases_lag2, cases_lag3])),
            "cases_roll3_max": float(np.max([cases_lag1, cases_lag2, cases_lag3])),
            "month_sin": float(np.sin(2 * np.pi * int(selected_month) / 12.0)),
            "month_cos": float(np.cos(2 * np.pi * int(selected_month) / 12.0)),
        }

        with st.expander("View generated model input features"):
            input_preview = pd.DataFrame([{k: features.get(k, 0.0) for k in feature_cols_from_meta}])
            st.dataframe(round_display_columns(input_preview, feature_cols_from_meta, 4), use_container_width=True)

        if st.button("Predict Outbreak", type="primary"):
            input_df = pd.DataFrame([{k: features.get(k, 0.0) for k in feature_cols_from_meta}])
            for col in input_df.columns:
                input_df[col] = pd.to_numeric(input_df[col], errors="coerce").fillna(0.0)

            try:
                pred = int(model.predict(input_df)[0])
                prob = get_outbreak_probability(model, input_df)

                col_a, col_b = st.columns(2)
                if pred == 1:
                    col_a.error("Prediction: OUTBREAK")
                else:
                    col_a.success("Prediction: NON-OUTBREAK")

                if not pd.isna(prob):
                    col_b.info(f"Outbreak Probability: {prob:.4f}")
                else:
                    col_b.info("Outbreak Probability: Not available")

                if meta.get("outbreak_definition"):
                    st.caption(meta.get("outbreak_definition"))
                else:
                    st.caption("Threshold for outbreak classification is based on the historical monthly case threshold used during model training.")
            except Exception as exc:
                st.error("Prediction failed. Check whether the model and feature columns match the exported artifacts.")
                st.exception(exc)

st.markdown("---")
st.caption("Baguio City Dengue Outbreak Forecast Dashboard | Powered by Machine Learning")
