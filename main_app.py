import json
import zipfile
from io import BytesIO
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Baguio City Dengue Forecast Dashboard",
    page_icon="🦟",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container {padding-top: 1.3rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.75rem;}
.small-note {font-size: 0.9rem; color: #666;}
.warning-note {font-size: 0.9rem; color: #9a5b00;}
</style>
""",
    unsafe_allow_html=True,
)

st.title("Baguio City Dengue Forecast Dashboard")
st.caption("Interactive dashboard for dengue outbreak forecasting using climate and epidemiological data.")


# =============================================================================
# CONFIG
# =============================================================================
APP_ROOT = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ARTIFACTS_DIR = APP_ROOT / "artifacts"

DEFAULT_FEATURE_COLS = [
    "rainfall",
    "relative_humidity",
    "temp_mid",
    "cases_lag_1",
    "cases_lag_2",
    "cases_lag_3",
    "rainfall_lag_1",
    "rainfall_lag_2",
    "rainfall_lag_3",
    "relative_humidity_lag_1",
    "relative_humidity_lag_2",
    "relative_humidity_lag_3",
    "temp_mid_lag_1",
    "temp_mid_lag_2",
    "temp_mid_lag_3",
    "cases_roll3_mean",
    "cases_roll3_max",
    "month_sin",
    "month_cos",
]

ARTIFACT_SPECS = {
    "monthly": ["monthly_modeling_dataset.csv", "monthly.csv"],
    "model_comparison": ["model_comparison.csv", "results_df.csv"],
    "auc_df": ["model_auc.csv", "auc_df.csv"],
    "feature_importance": ["feature_importance.csv", "importance_df.csv"],
    "feature_sensitivity": ["feature_sensitivity.csv", "sensitivity_df.csv"],
    "forecast": ["forecast_5yr.csv", "forecast_df.csv"],
    "barangay_monthly": ["barangay_monthly.csv"],
    "top_barangay_monthly": ["top_barangay_monthly.csv", "monthly_top_barangay.csv"],
    "top3_barangays_yearly": ["top3_barangays_yearly.csv"],
    "top3_barangays_overall": ["top3_barangays_overall.csv"],
    "test_predictions": ["test_predictions.csv"],
    "climate_case_correlation": ["climate_case_correlation.csv"],
    "month_profile": ["month_profile.csv"],
    "forecast_barangay_ranking": ["forecast_barangay_ranking.csv"],
    "forecast_top3_barangays": ["forecast_top3_barangays.csv"],
    "barangay_risk_profile": ["barangay_risk_profile.csv"],
}

DATE_KEYS = {
    "monthly",
    "forecast",
    "barangay_monthly",
    "top_barangay_monthly",
    "test_predictions",
    "forecast_barangay_ranking",
    "forecast_top3_barangays",
}

JSON_SPECS = {
    "meta": ["meta.json"],
    "feature_columns": ["feature_columns.json"],
}

MODEL_NAMES = ["best_model.joblib", "best_model.pkl"]


# =============================================================================
# ARTIFACT LOADING HELPERS
# =============================================================================
def dashboard_zip_candidates():
    """Find dashboard_artifacts ZIP files placed beside main_app.py."""
    candidates = []
    for pattern in ["dashboard_artifacts.zip", "dashboard_artifacts*.zip"]:
        candidates.extend(APP_ROOT.glob(pattern))
    # Preserve order and remove duplicates.
    seen = set()
    unique = []
    for path in candidates:
        if path.exists() and path.is_file() and path not in seen:
            seen.add(path)
            unique.append(path)
    return unique


def possible_member_names(filename):
    return [
        f"artifacts/{filename}",
        filename,
        f"./artifacts/{filename}",
        f"./{filename}",
    ]


def find_zip_member(zf, file_names):
    """Return the matching member in a ZIP, allowing artifacts/ prefix."""
    members = [m for m in zf.namelist() if not m.endswith("/")]
    lower_to_original = {m.lower(): m for m in members}

    for file_name in file_names:
        for candidate in possible_member_names(file_name):
            hit = lower_to_original.get(candidate.lower())
            if hit:
                return hit

    # Fallback: match by basename in case the ZIP has a different folder name.
    wanted = {Path(name).name.lower() for name in file_names}
    for member in members:
        if Path(member).name.lower() in wanted:
            return member
    return None


def read_local_bytes(file_names):
    for file_name in file_names:
        for base in [ARTIFACTS_DIR, APP_ROOT]:
            path = base / file_name
            if path.exists() and path.is_file():
                return path.read_bytes(), str(path.relative_to(APP_ROOT))
    return None, None


def read_zip_bytes(file_names, uploaded_zip_bytes=None):
    # Uploaded ZIP has priority over ZIPs already saved in the project.
    zip_sources = []
    if uploaded_zip_bytes:
        zip_sources.append(("uploaded dashboard ZIP", uploaded_zip_bytes))
    for zip_path in dashboard_zip_candidates():
        try:
            zip_sources.append((zip_path.name, zip_path.read_bytes()))
        except OSError:
            pass

    for source_name, zip_bytes in zip_sources:
        try:
            with zipfile.ZipFile(BytesIO(zip_bytes)) as zf:
                member = find_zip_member(zf, file_names)
                if member:
                    return zf.read(member), f"{source_name}:{member}"
        except zipfile.BadZipFile:
            continue
    return None, None


def read_artifact_bytes(file_names, uploaded_zip_bytes=None):
    # Uploaded ZIP should override saved files so users can test a newer artifact package.
    if uploaded_zip_bytes:
        zip_bytes, zip_source = read_zip_bytes(file_names, uploaded_zip_bytes)
        if zip_bytes is not None:
            return zip_bytes, zip_source

    local_bytes, local_source = read_local_bytes(file_names)
    if local_bytes is not None:
        return local_bytes, local_source

    return read_zip_bytes(file_names, None)


def read_csv_artifact(file_names, uploaded_zip_bytes=None):
    raw, source = read_artifact_bytes(file_names, uploaded_zip_bytes)
    if raw is None:
        return None, None
    return pd.read_csv(BytesIO(raw)), source


def read_json_artifact(file_names, uploaded_zip_bytes=None):
    raw, source = read_artifact_bytes(file_names, uploaded_zip_bytes)
    if raw is None:
        return None, None
    return json.loads(raw.decode("utf-8")), source


def load_joblib_artifact(file_names, uploaded_zip_bytes=None):
    raw, source = read_artifact_bytes(file_names, uploaded_zip_bytes)
    if raw is None:
        return None, None
    return joblib.load(BytesIO(raw)), source


def load_artifacts(uploaded_zip_bytes=None):
    loaded = {}
    sources = {}

    for key, file_names in ARTIFACT_SPECS.items():
        df, source = read_csv_artifact(file_names, uploaded_zip_bytes)
        loaded[key] = df
        if source:
            sources[key] = source

    for key, file_names in JSON_SPECS.items():
        data, source = read_json_artifact(file_names, uploaded_zip_bytes)
        loaded[key] = data
        if source:
            sources[key] = source

    return loaded, sources


@st.cache_resource(show_spinner=False)
def load_model(uploaded_zip_bytes=None):
    model_obj, source = load_joblib_artifact(MODEL_NAMES, uploaded_zip_bytes)
    return model_obj, source


# =============================================================================
# GENERAL HELPERS
# =============================================================================
def month_name_from_number(month_num):
    month_names = {
        1: "Jan",
        2: "Feb",
        3: "Mar",
        4: "Apr",
        5: "May",
        6: "Jun",
        7: "Jul",
        8: "Aug",
        9: "Sep",
        10: "Oct",
        11: "Nov",
        12: "Dec",
    }
    try:
        return month_names.get(int(month_num), str(month_num))
    except Exception:
        return str(month_num)


def safe_to_float(value, default=np.nan):
    try:
        out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        return float(out) if pd.notna(out) else float(default)
    except Exception:
        return float(default)


def safe_metric_value(value, decimals=2):
    value = safe_to_float(value, np.nan)
    if pd.isna(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def round_display_columns(df, columns, decimals=2):
    if df is None:
        return None
    display_df = df.copy()
    for col in columns:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(decimals)
    return display_df


def parse_date_columns(dfs):
    parsed = {}
    for key, value in dfs.items():
        if isinstance(value, pd.DataFrame):
            df = value.copy()
            if key in DATE_KEYS and "Date" in df.columns:
                df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            parsed[key] = df
        else:
            parsed[key] = value
    return parsed


def display_clean_barangay_columns(df):
    if df is None:
        return None
    rename_map = {
        "Top_Barangay": "Barangay",
        "Top_Barangay_Cases": "Barangay_cases",
        "rank_within_year": "Rank Within Year",
        "rank": "Rank",
        "Barangay_cases": "Dengue Cases",
    }
    return df.copy().rename(columns={k: v for k, v in rename_map.items() if k in df.columns})


def complete_month_profile(month_profile_df, monthly_df):
    """Ensure month_profile has MonthName, CHSO case, and climate columns."""
    if monthly_df is None or "Month" not in monthly_df.columns:
        return month_profile_df

    needed = ["CHSO_cases", "rainfall", "relative_humidity", "temp_mid"]
    available_needed = [col for col in needed if col in monthly_df.columns]
    if not available_needed:
        return month_profile_df

    generated = monthly_df.groupby("Month", as_index=False)[available_needed].mean(numeric_only=True)
    generated["MonthName"] = generated["Month"].apply(month_name_from_number)

    if month_profile_df is None or month_profile_df.empty or "Month" not in month_profile_df.columns:
        return generated

    fixed = month_profile_df.copy()
    for col in needed:
        if col not in fixed.columns and col in generated.columns:
            fixed = fixed.merge(generated[["Month", col]], on="Month", how="left")
    if "MonthName" not in fixed.columns:
        fixed["MonthName"] = fixed["Month"].apply(month_name_from_number)
    return fixed


def build_climate_case_correlation(monthly_df):
    if monthly_df is None or "CHSO_cases" not in monthly_df.columns:
        return None
    rows = []
    for feature in ["rainfall", "relative_humidity", "temp_mid"]:
        if feature in monthly_df.columns:
            sub = monthly_df[[feature, "CHSO_cases"]].apply(pd.to_numeric, errors="coerce").dropna()
            corr = sub.corr().iloc[0, 1] if len(sub) > 1 else np.nan
            rows.append({"feature": feature, "pearson_corr_with_CHSO_cases": corr})
    if not rows:
        return None
    return pd.DataFrame(rows).sort_values("pearson_corr_with_CHSO_cases", ascending=False)


def normalize_auc_table(auc_table):
    if auc_table is None or auc_table.empty:
        return auc_table
    df = auc_table.copy()
    df = df.rename(columns={"Model": "model", "AUC": "auc", "Auc": "auc"})
    return df


def normalize_model_comparison(model_df, auc_table=None):
    if model_df is None or model_df.empty:
        return model_df

    df = model_df.copy()
    df = df.rename(
        columns={
            "Model": "model",
            "Accuracy": "accuracy",
            "F1 Score": "f1_score",
            "Precision": "precision",
            "Recall": "recall",
            "Reliability (Brier)": "brier",
            "AUC (Supplementary)": "auc",
            "AUC": "auc",
        }
    )

    if "model" not in df.columns:
        return df

    if "auc" not in df.columns and auc_table is not None and not auc_table.empty:
        auc_tmp = normalize_auc_table(auc_table)
        if auc_tmp is not None and {"model", "auc"}.issubset(auc_tmp.columns):
            df = df.merge(auc_tmp[["model", "auc"]], on="model", how="left")

    for col in ["accuracy", "f1_score", "precision", "recall", "brier", "auc"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def normalize_feature_sensitivity(feature_sensitivity_df):
    if feature_sensitivity_df is None or feature_sensitivity_df.empty:
        return feature_sensitivity_df

    df = feature_sensitivity_df.copy()

    df = df.rename(
        columns={
            # Base probability
            "base_avg_outbreak_prob": "base_avg_outbreak_probability",
            "base_avg_outbreak_probability": "base_avg_outbreak_probability",

            # New probability after +10% increase
            "new_avg_outbreak_proba": "new_avg_outbreak_probability",
            "new_avg_outbreak_probability (10% increase)":
                "new_avg_outbreak_probability",

            # Change in probability
            "change_in_probability": "delta_probability",

            # Percentage change
            "percent_change": "percent_change",
        }
    )

    return df

def get_profile_value(month_num, col_name, month_profile_df=None, fallback_df=None, default=0.0):
    if month_profile_df is not None and not month_profile_df.empty and "Month" in month_profile_df.columns:
        subset = month_profile_df[pd.to_numeric(month_profile_df["Month"], errors="coerce") == int(month_num)]
        if not subset.empty and col_name in subset.columns:
            value = safe_to_float(subset.iloc[0][col_name], np.nan)
            if pd.notna(value):
                return value

    if fallback_df is not None and col_name in fallback_df.columns:
        series = pd.to_numeric(fallback_df[col_name], errors="coerce").dropna()
        if not series.empty:
            return float(series.mean())
    return float(default)


def get_reasonable_range(df, col_name, fallback_min=0.0, fallback_max=100.0):
    if df is not None and col_name in df.columns:
        series = pd.to_numeric(df[col_name], errors="coerce").dropna()
        if not series.empty:
            vmin = float(series.min())
            vmax = float(series.max())
            if vmin == vmax:
                vmax = vmin + 1.0
            return vmin, vmax
    return float(fallback_min), float(fallback_max)


def get_forecast_row(forecast_df, year_num, month_num):
    if forecast_df is None or forecast_df.empty or not {"Year", "Month"}.issubset(forecast_df.columns):
        return None
    subset = forecast_df[
        (pd.to_numeric(forecast_df["Year"], errors="coerce") == int(year_num))
        & (pd.to_numeric(forecast_df["Month"], errors="coerce") == int(month_num))
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def previous_month_numbers(year_num, month_num):
    periods = []
    current_year = int(year_num)
    current_month = int(month_num)
    for _ in range(3):
        current_month -= 1
        if current_month == 0:
            current_month = 12
            current_year -= 1
        periods.append((current_year, current_month))
    return periods


def build_live_prediction_features(
    year_num,
    month_num,
    rainfall_now,
    humidity_now,
    temp_now,
    cases_lag_1,
    cases_lag_2,
    cases_lag_3,
    month_profile_df,
    monthly_df,
    forecast_row=None,
):
    if forecast_row is not None:
        rainfall_lag_1 = safe_to_float(forecast_row.get("rainfall_lag_1", 0.0), 0.0)
        rainfall_lag_2 = safe_to_float(forecast_row.get("rainfall_lag_2", 0.0), 0.0)
        rainfall_lag_3 = safe_to_float(forecast_row.get("rainfall_lag_3", 0.0), 0.0)

        rh_lag_1 = safe_to_float(forecast_row.get("relative_humidity_lag_1", 0.0), 0.0)
        rh_lag_2 = safe_to_float(forecast_row.get("relative_humidity_lag_2", 0.0), 0.0)
        rh_lag_3 = safe_to_float(forecast_row.get("relative_humidity_lag_3", 0.0), 0.0)

        temp_lag_1 = safe_to_float(forecast_row.get("temp_mid_lag_1", 0.0), 0.0)
        temp_lag_2 = safe_to_float(forecast_row.get("temp_mid_lag_2", 0.0), 0.0)
        temp_lag_3 = safe_to_float(forecast_row.get("temp_mid_lag_3", 0.0), 0.0)
    else:
        (_, prev1), (_, prev2), (_, prev3) = previous_month_numbers(year_num, month_num)
        rainfall_lag_1 = get_profile_value(prev1, "rainfall", month_profile_df, monthly_df, 0.0)
        rainfall_lag_2 = get_profile_value(prev2, "rainfall", month_profile_df, monthly_df, 0.0)
        rainfall_lag_3 = get_profile_value(prev3, "rainfall", month_profile_df, monthly_df, 0.0)
        rh_lag_1 = get_profile_value(prev1, "relative_humidity", month_profile_df, monthly_df, 0.0)
        rh_lag_2 = get_profile_value(prev2, "relative_humidity", month_profile_df, monthly_df, 0.0)
        rh_lag_3 = get_profile_value(prev3, "relative_humidity", month_profile_df, monthly_df, 0.0)
        temp_lag_1 = get_profile_value(prev1, "temp_mid", month_profile_df, monthly_df, 0.0)
        temp_lag_2 = get_profile_value(prev2, "temp_mid", month_profile_df, monthly_df, 0.0)
        temp_lag_3 = get_profile_value(prev3, "temp_mid", month_profile_df, monthly_df, 0.0)

    cases_lag_values = [
        safe_to_float(cases_lag_1, 0.0),
        safe_to_float(cases_lag_2, 0.0),
        safe_to_float(cases_lag_3, 0.0),
    ]

    return {
        "rainfall": float(rainfall_now),
        "relative_humidity": float(humidity_now),
        "temp_mid": float(temp_now),
        "cases_lag_1": cases_lag_values[0],
        "cases_lag_2": cases_lag_values[1],
        "cases_lag_3": cases_lag_values[2],
        "rainfall_lag_1": float(rainfall_lag_1),
        "rainfall_lag_2": float(rainfall_lag_2),
        "rainfall_lag_3": float(rainfall_lag_3),
        "relative_humidity_lag_1": float(rh_lag_1),
        "relative_humidity_lag_2": float(rh_lag_2),
        "relative_humidity_lag_3": float(rh_lag_3),
        "temp_mid_lag_1": float(temp_lag_1),
        "temp_mid_lag_2": float(temp_lag_2),
        "temp_mid_lag_3": float(temp_lag_3),
        "cases_roll3_mean": float(np.mean(cases_lag_values)),
        "cases_roll3_max": float(np.max(cases_lag_values)),
        "month_sin": float(np.sin(2 * np.pi * int(month_num) / 12.0)),
        "month_cos": float(np.cos(2 * np.pi * int(month_num) / 12.0)),
    }


def outbreak_label_from_binary(x):
    try:
        return "Outbreak" if int(x) == 1 else "Non-outbreak"
    except Exception:
        return "Unknown"


def select_numeric_columns(df, preferred):
    return [col for col in preferred if col in df.columns]


def get_feature_columns(meta, feature_columns_json, model_obj):
    if isinstance(meta, dict) and meta.get("feature_columns"):
        return list(meta["feature_columns"])
    if isinstance(feature_columns_json, list) and feature_columns_json:
        return list(feature_columns_json)
    if model_obj is not None and hasattr(model_obj, "feature_names_in_"):
        return list(model_obj.feature_names_in_)
    return DEFAULT_FEATURE_COLS

def interpret_metric(metric, value):
    value = safe_to_float(value, np.nan)

    if pd.isna(value):
        return "N/A"

    metric = str(metric).lower()

    if metric == "accuracy":
        if value >= 0.90:
            return "Excellent"
        elif value >= 0.80:
            return "Good"
        elif value >= 0.70:
            return "Moderate"
        return "Weak"

    if metric == "precision":
        if value >= 0.90:
            return "Excellent"
        elif value >= 0.70:
            return "Good"
        elif value >= 0.65:
            return "Moderate"
        return "Weak"

    if metric == "recall":
        if value >= 0.90:
            return "Excellent"
        elif value >= 0.70:
            return "Good"
        elif value >= 0.50:
            return "Moderate"
        return "Weak"

    if metric == "f1_score":
        if value >= 0.90:
            return "Excellent"
        elif value >= 0.80:
            return "Good"
        elif value >= 0.70:
            return "Moderate"
        return "Weak"

    if metric == "auc":
        if value >= 0.90:
            return "Excellent"
        elif value >= 0.70:
            return "Acceptable"
        elif value >= 0.50:
            return "Weak"
        return "Below Chance"

    if metric == "brier":
        if value <= 0.09:
            return "Strongest"
        elif value <= 0.10:
            return "Strong"
        elif value <= 0.20:
            return "Moderate"
        return "Weak"

    return "Not Interpreted"

def build_metric_guide_df():
    return pd.DataFrame(
        [
            [
                "Accuracy",
                "0.90–1.00",
                "0.80–0.89",
                "0.70–0.79",
                "Below 0.70",
                "Higher is better",
            ],
            [
                "Precision",
                "0.90–1.00",
                "0.70–0.89",
                "0.65–0.69",
                "Below 0.65",
                "Higher is better",
            ],
            [
                "Recall",
                "0.90–1.00",
                "0.70–0.89",
                "0.50–0.69",
                "Below 0.50",
                "Higher is better",
            ],
            [
                "F1 Score",
                "0.90–1.00",
                "0.80–0.89",
                "0.70–0.79",
                "Below 0.70",
                "Higher is better",
            ],
            [
                "AUC",
                "0.90–1.00",
                "0.70–0.89",
                "0.50–0.69",
                "Below 0.50",
                "Higher is better",
            ],
            [
                "Brier Score",
                "0.00–0.09",
                ">0.09–0.10",
                ">0.10–0.20",
                "Above 0.20",
                "Lower is better",
            ],
        ],
        columns=[
            "Metric",
            "Excellent / Strongest",
            "Good / Strong",
            "Moderate / Acceptable",
            "Weak / Below Chance",
            "Direction",
        ],
    )

# =============================================================================
# SIDEBAR: LOAD ZIP OR FOLDER
# =============================================================================
st.sidebar.header("Dashboard Files")
st.sidebar.caption(
    "The app can read files from an `artifacts/` folder, from `dashboard_artifacts.zip`, or from the uploader below."
)

uploaded_zip = st.sidebar.file_uploader(
    "Upload dashboard_artifacts.zip",
    type=["zip"],
    help="Use this when the artifacts folder is not already saved with the Streamlit app.",
)
uploaded_zip_bytes = uploaded_zip.getvalue() if uploaded_zip is not None else None

artifacts, artifact_sources = load_artifacts(uploaded_zip_bytes)
model, model_source = load_model(uploaded_zip_bytes)
artifacts = parse_date_columns(artifacts)

with st.sidebar.expander("Manual individual file upload", expanded=False):
    manual_uploads = {
        "monthly": st.file_uploader("monthly_modeling_dataset.csv", type=["csv"], key="up_monthly"),
        "model_comparison": st.file_uploader("model_comparison.csv", type=["csv"], key="up_model_comparison"),
        "auc_df": st.file_uploader("model_auc.csv", type=["csv"], key="up_auc"),
        "feature_importance": st.file_uploader("feature_importance.csv", type=["csv"], key="up_importance"),
        "feature_sensitivity": st.file_uploader("feature_sensitivity.csv", type=["csv"], key="up_sensitivity"),
        "forecast": st.file_uploader("forecast_5yr.csv", type=["csv"], key="up_forecast"),
        "forecast_top3_barangays": st.file_uploader("forecast_top3_barangays.csv", type=["csv"], key="up_forecast_top3"),
        "barangay_risk_profile": st.file_uploader("barangay_risk_profile.csv", type=["csv"], key="up_risk"),
        "barangay_monthly": st.file_uploader("barangay_monthly.csv", type=["csv"], key="up_barangay_monthly"),
        "top_barangay_monthly": st.file_uploader("top_barangay_monthly.csv", type=["csv"], key="up_top_monthly"),
        "top3_barangays_yearly": st.file_uploader("top3_barangays_yearly.csv", type=["csv"], key="up_top3_yearly"),
        "top3_barangays_overall": st.file_uploader("top3_barangays_overall.csv", type=["csv"], key="up_top3_overall"),
        "test_predictions": st.file_uploader("test_predictions.csv", type=["csv"], key="up_test_predictions"),
        "climate_case_correlation": st.file_uploader("climate_case_correlation.csv", type=["csv"], key="up_correlation"),
        "month_profile": st.file_uploader("month_profile.csv", type=["csv"], key="up_month_profile"),
    }
    uploaded_meta = st.file_uploader("meta.json", type=["json"], key="up_meta")
    uploaded_feature_columns = st.file_uploader("feature_columns.json", type=["json"], key="up_feature_columns")
    uploaded_model = st.file_uploader("best_model.joblib", type=["joblib", "pkl"], key="up_model")

for key, file_obj in manual_uploads.items():
    if file_obj is not None:
        artifacts[key] = pd.read_csv(file_obj)
        artifact_sources[key] = "manual upload"

if uploaded_meta is not None:
    artifacts["meta"] = json.load(uploaded_meta)
    artifact_sources["meta"] = "manual upload"

if uploaded_feature_columns is not None:
    artifacts["feature_columns"] = json.load(uploaded_feature_columns)
    artifact_sources["feature_columns"] = "manual upload"

if uploaded_model is not None:
    model = joblib.load(uploaded_model)
    model_source = "manual upload"

artifacts = parse_date_columns(artifacts)

monthly = artifacts.get("monthly")
model_comparison = artifacts.get("model_comparison")
auc_df = normalize_auc_table(artifacts.get("auc_df"))
feature_importance = artifacts.get("feature_importance")
feature_sensitivity = normalize_feature_sensitivity(artifacts.get("feature_sensitivity"))
forecast = artifacts.get("forecast")
barangay_monthly = artifacts.get("barangay_monthly")
top_barangay_monthly = artifacts.get("top_barangay_monthly")
top3_barangays_yearly = artifacts.get("top3_barangays_yearly")
top3_barangays_overall = artifacts.get("top3_barangays_overall")
test_predictions = artifacts.get("test_predictions")
climate_case_correlation = artifacts.get("climate_case_correlation")
month_profile = artifacts.get("month_profile")
forecast_barangay_ranking = artifacts.get("forecast_barangay_ranking")
forecast_top3_barangays = artifacts.get("forecast_top3_barangays")
barangay_risk_profile = artifacts.get("barangay_risk_profile")
meta = artifacts.get("meta")
feature_columns_json = artifacts.get("feature_columns")

if monthly is None or monthly.empty:
    st.error(
        "Required file missing: `monthly_modeling_dataset.csv`. Put the exported `artifacts/` folder beside `main_app.py`, "
        "upload `dashboard_artifacts.zip` in the sidebar, or upload the CSV manually."
    )
    with st.expander("Expected artifact files"):
        st.write(sorted({name for names in ARTIFACT_SPECS.values() for name in names}))
    st.stop()

# Final data fixes after loading.
month_profile = complete_month_profile(month_profile, monthly)
model_comparison = normalize_model_comparison(model_comparison, auc_df)
if climate_case_correlation is None or climate_case_correlation.empty:
    climate_case_correlation = build_climate_case_correlation(monthly)
feature_columns = get_feature_columns(meta, feature_columns_json, model)

# Sidebar status.
if meta:
    st.sidebar.success(f"Best Model: {meta.get('best_model', 'Unknown')}")
    threshold_val = meta.get("outbreak_threshold_cases", meta.get("threshold", "N/A"))
    if isinstance(threshold_val, (int, float)):
        st.sidebar.info(f"Outbreak Threshold: {threshold_val:.2f} cases")
    else:
        st.sidebar.info(f"Outbreak Threshold: {threshold_val}")
else:
    st.sidebar.warning("meta.json not found. The dashboard can still run with the required CSV files.")

if model is not None:
    st.sidebar.success(f"Model loaded: {model_source or 'available'}")
else:
    st.sidebar.warning("best_model.joblib not loaded. Live prediction will be disabled.")

with st.sidebar.expander("Loaded artifact status", expanded=False):
    status_rows = []
    for key in list(ARTIFACT_SPECS.keys()) + ["meta", "feature_columns"]:
        value = artifacts.get(key)
        loaded = value is not None and not (isinstance(value, pd.DataFrame) and value.empty)
        status_rows.append(
            {
                "artifact": key,
                "status": "Loaded" if loaded else "Missing",
                "source": artifact_sources.get(key, ""),
            }
        )
    st.dataframe(pd.DataFrame(status_rows), use_container_width=True, hide_index=True)


# =============================================================================
# TABS
# =============================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Overview",
        "Barangay Analytics",
        "Model Results",
        "Feature Transparency",
        "Forecast & Live Prediction",
    ]
)


# =============================================================================
# TAB 1: OVERVIEW
# =============================================================================
with tab1:
    st.header("Historical Dengue Overview")

    total_months = len(monthly)
    total_cases = (
        int(pd.to_numeric(monthly["CHSO_cases"], errors="coerce").fillna(0).sum())
        if "CHSO_cases" in monthly.columns
        else 0
    )
    avg_cases = pd.to_numeric(monthly["CHSO_cases"], errors="coerce").mean() if "CHSO_cases" in monthly.columns else np.nan
    outbreak_months = (
        int(pd.to_numeric(monthly["outbreak"], errors="coerce").fillna(0).sum())
        if "outbreak" in monthly.columns
        else "N/A"
    )

    metric1, metric2, metric3, metric4 = st.columns(4)
    metric1.metric("Total Months", total_months)
    metric2.metric("Total CHSO Cases", f"{total_cases:,}")
    metric3.metric("Average Monthly Cases", safe_metric_value(avg_cases))
    metric4.metric("Outbreak Months", outbreak_months)

    st.subheader("Model Prediction Target")
    if meta:
        st.info(
            f"**Problem Definition:** {meta.get('problem_definition', 'Monthly outbreak classification')}  \n"
            f"**Outbreak Definition:** {meta.get('outbreak_definition', 'Monthly CHSO cases greater than or equal to the selected threshold.')}"
        )
    else:
        st.info("The model predicts whether a selected month is classified as an outbreak or non-outbreak month.")

    st.subheader("Monthly Dengue Cases")
    if {"Date", "CHSO_cases"}.issubset(monthly.columns):
        trend_cols = ["Date", "CHSO_cases"]
        if "DOH_cases" in monthly.columns:
            trend_cols.append("DOH_cases")
            trend_long = monthly[trend_cols].melt(id_vars="Date", var_name="Source", value_name="Cases")
            fig_line = px.line(
                trend_long,
                x="Date",
                y="Cases",
                color="Source",
                markers=True,
                title="Monthly Dengue Cases: CHSO and DOH Comparison",
                color_discrete_map={
                    "CHSO_cases": "#AE75DA",
                    "DOH_cases": "#4382DF"   
                }
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
    else:
        st.warning("Monthly trend requires Date and CHSO_cases columns.")

    st.subheader("Year-Month Heatmap of CHSO Dengue Cases")
    if {"Year", "Month", "CHSO_cases"}.issubset(monthly.columns):
        heat = monthly.pivot_table(index="Year", columns="Month", values="CHSO_cases", aggfunc="sum")
        fig_heat = px.imshow(
            heat,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="YlOrRd",
            title="Year-Month Heatmap of CHSO Dengue Cases",
        )
        fig_heat.update_xaxes(title="Month")
        fig_heat.update_yaxes(title="Year")
        st.plotly_chart(fig_heat, use_container_width=True)
    else:
        st.warning("Heatmap requires Year, Month, and CHSO_cases columns.")

    st.subheader("Climate-Case Correlation")
    fig_corr = px.bar(climate_case_correlation, x="feature", y="pearson_corr_with_CHSO_cases", text="pearson_corr_with_CHSO_cases")
    st.plotly_chart(fig_corr, use_container_width=True)
    st.dataframe(climate_case_correlation, use_container_width=True)

    st.subheader("Average Monthly Profile")
    fig_month = px.bar(month_profile, x="MonthName", y="CHSO_cases", text="CHSO_cases")
    st.plotly_chart(fig_month, use_container_width=True)
    st.dataframe(month_profile, use_container_width=True)

    st.subheader("Climate Profile of Outbreak vs Non-outbreak Months")
    if {"outbreak", "rainfall", "relative_humidity", "temp_mid"}.issubset(monthly.columns):
        climate_profile = monthly.groupby("outbreak", as_index=False)[["rainfall", "relative_humidity", "temp_mid"]].mean(numeric_only=True)
        climate_profile["Outbreak Status"] = climate_profile["outbreak"].map({0: "Non-outbreak", 1: "Outbreak"})
        climate_long = climate_profile.melt(
            id_vars="Outbreak Status",
            value_vars=["rainfall", "relative_humidity", "temp_mid"],
            var_name="Climate Variable",
            value_name="Average Value",
        )
        climate_long = round_display_columns(climate_long, ["Average Value"], 2)
        fig_climate_profile = px.bar(
            climate_long,
            x="Climate Variable",
            y="Average Value",
            color="Outbreak Status",
            barmode="group",
            text="Average Value",
            title="Climate Profile of Outbreak vs Non-outbreak Months",
        )
        fig_climate_profile.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_climate_profile, use_container_width=True)
    else:
        st.caption("Climate profile is shown when monthly_modeling_dataset.csv includes outbreak and climate columns.")


# =============================================================================
# TAB 2: BARANGAY ANALYTICS
# =============================================================================
with tab2:
    st.header("Barangay Analytics")

    st.subheader("Barangay with the Highest Monthly Dengue Cases")
    if top_barangay_monthly is not None and not top_barangay_monthly.empty:
        st.dataframe(display_clean_barangay_columns(top_barangay_monthly), use_container_width=True, hide_index=True)
    else:
        st.warning("top_barangay_monthly.csv is unavailable.")

    st.subheader("Barangays with the Highest Dengue Cases")
    ranking_choice = st.radio(
        "Choose ranking view",
        ["Three Highest per Year", "Three Highest Overall"],
        horizontal=True,
    )

    if ranking_choice == "Three Highest per Year":
        if top3_barangays_yearly is not None and not top3_barangays_yearly.empty:
            yearly_display = display_clean_barangay_columns(top3_barangays_yearly)
            st.dataframe(yearly_display, use_container_width=True, hide_index=True)
            if {"Year", "Barangay", "Barangay_cases"}.issubset(top3_barangays_yearly.columns):
                fig_tree = px.treemap(
                    top3_barangays_yearly,
                    path=["Year", "Barangay"],
                    values="Barangay_cases",
                    color="Barangay_cases",
                    color_continuous_scale="Plasma",
                    title="Three Barangays with the Highest Dengue Cases per Year",
                    height=650,
                )
                st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.warning("top3_barangays_yearly.csv is unavailable.")
    else:
        if top3_barangays_overall is not None and not top3_barangays_overall.empty:
            overall_display = display_clean_barangay_columns(top3_barangays_overall)
            st.dataframe(overall_display, use_container_width=True, hide_index=True)
            if {"Barangay", "Barangay_cases"}.issubset(top3_barangays_overall.columns):
                fig_overall = px.bar(
                    top3_barangays_overall,
                    x="Barangay",
                    y="Barangay_cases",
                    text="Barangay_cases",
                    title="Three Barangays with the Highest Overall Dengue Cases",
                )
                fig_overall.update_traces(textposition="outside")
                st.plotly_chart(fig_overall, use_container_width=True)
        else:
            st.warning("top3_barangays_overall.csv is unavailable.")

    st.subheader("Barangay Monthly Records")
    if barangay_monthly is not None and not barangay_monthly.empty:
        barangay_view = barangay_monthly.copy()

        filter_col1, filter_col2 = st.columns(2)
        if "Year" in barangay_view.columns:
            years = sorted(pd.to_numeric(barangay_view["Year"], errors="coerce").dropna().astype(int).unique().tolist())
            selected_years = filter_col1.multiselect("Filter year", years, default=years)
            if selected_years:
                barangay_view = barangay_view[pd.to_numeric(barangay_view["Year"], errors="coerce").isin(selected_years)]
        if "Barangay" in barangay_view.columns:
            barangays = sorted(barangay_view["Barangay"].dropna().astype(str).unique().tolist())
            selected_barangays = filter_col2.multiselect("Filter barangay", barangays, default=[])
            if selected_barangays:
                barangay_view = barangay_view[barangay_view["Barangay"].astype(str).isin(selected_barangays)]

        st.dataframe(display_clean_barangay_columns(barangay_view), use_container_width=True, hide_index=True)
    else:
        st.warning("barangay_monthly.csv is unavailable.")


# =============================================================================
# TAB 3: MODEL RESULTS
# =============================================================================
with tab3:
    st.header("Model Results")

    if meta:
        st.success(f"Selected Model: {meta.get('best_model', 'Unknown')}")

    if model_comparison is not None and not model_comparison.empty:
        display_cols = [
            col
            for col in ["model", "accuracy", "f1_score", "precision", "recall", "auc", "brier"]
            if col in model_comparison.columns
        ]
        numeric_display_cols = [col for col in display_cols if col != "model"]
        display_df = round_display_columns(model_comparison[display_cols], numeric_display_cols, 4)
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        metric_cols = [
            col
            for col in [
                "accuracy",
                "f1_score",
                "precision",
                "recall",
                "auc",
            ]
            if col in model_comparison.columns
        ]
        
        if "model" in model_comparison.columns and metric_cols:
            st.subheader("Model Comparison by Classification Metric")
        
            results_long = model_comparison.melt(
                id_vars="model",
                value_vars=metric_cols,
                var_name="Metric",
                value_name="Score",
            )
        
            results_long["Score"] = pd.to_numeric(
                results_long["Score"],
                errors="coerce",
            )
        
            metric_name_map = {
                "accuracy": "Accuracy",
                "precision": "Precision",
                "recall": "Recall",
                "f1_score": "F1 Score",
                "auc": "AUC",
            }
        
            results_long["Metric Label"] = (
                results_long["Metric"]
                .map(metric_name_map)
                .fillna(results_long["Metric"])
            )
        
            results_long["Rating"] = results_long.apply(
                lambda row: interpret_metric(
                    row["Metric"],
                    row["Score"],
                ),
                axis=1,
            )
        
            results_long["Hover Result"] = results_long.apply(
                lambda row: (
                    "N/A"
                    if pd.isna(row["Score"])
                    else f"{row['Score']:.3f} — {row['Rating']}"
                ),
                axis=1,
            )
        
            fig_model = px.bar(
                results_long,
                x="model",
                y="Score",
                color="Metric Label",
                barmode="group",
                text="Score",
                custom_data=[
                    "Metric Label",
                    "Hover Result",
                ],
                title="Model Comparison by Metric",
            )
        
            fig_model.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
                hovertemplate=(
                    "<b>Model:</b> %{x}<br>"
                    "<b>Metric:</b> %{customdata[0]}<br>"
                    "<b>Result:</b> %{customdata[1]}<br>"
                    "<b>Direction:</b> Higher is better"
                    "<extra></extra>"
                ),
            )
        
            fig_model.update_yaxes(
                range=[0, 1.15],
                title="Score",
            )
        
            st.plotly_chart(
                fig_model,
                use_container_width=True,
            )

        if {"model", "brier"}.issubset(model_comparison.columns):
            st.subheader("Reliability Score")
        
            brier_display = model_comparison[
                ["model", "brier"]
            ].copy()
        
            brier_display["brier"] = pd.to_numeric(
                brier_display["brier"],
                errors="coerce",
            )
        
            brier_display["Rating"] = brier_display[
                "brier"
            ].apply(
                lambda value: interpret_metric("brier", value)
            )
        
            brier_display["Hover Result"] = brier_display.apply(
                lambda row: (
                    "N/A"
                    if pd.isna(row["brier"])
                    else (
                        f"{row['brier']:.4f} — "
                        f"{row['Rating']}"
                    )
                ),
                axis=1,
            )
        
            fig_brier = px.bar(
                brier_display,
                x="model",
                y="brier",
                text="brier",
                custom_data=["Hover Result"],
                title="Reliability (Brier Score: lower is better)",
            )
        
            fig_brier.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
                hovertemplate=(
                    "<b>Model:</b> %{x}<br>"
                    "<b>Metric:</b> Brier Score<br>"
                    "<b>Result:</b> %{customdata[0]}<br>"
                    "<b>Direction:</b> Lower is better"
                    "<extra></extra>"
                ),
            )
        
            st.plotly_chart(
                fig_brier,
                use_container_width=True,
            )
    else:
        st.warning("model_comparison.csv is unavailable.")

    st.subheader("How to Read the Metrics")
    st.markdown(
        """
    - **Accuracy** – Measures the overall percentage of correct classifications.
    - **Precision** – Measures how often predicted outbreak months are actual outbreaks.
    - **Recall/Sensitivity** – Measures how many actual outbreak months the model correctly detects.
    - **F1 Score** – Balances precision and recall in one metric.
    - **AUC** – Measures how well the model separates outbreak from non-outbreak months across different decision thresholds.
    - **Brier Score** – Measures the reliability of predicted probabilities; a lower score is better.
    """
    )

    st.subheader(
        "General Guide for Interpreting Model Performance Metrics"
    )
    
    st.caption(
        "These ranges are a practical dashboard guide, not "
        "universal official cutoffs. They are aligned with the "
        "labels displayed in the result table and graph tooltips."
    )
    
    st.dataframe(
        build_metric_guide_df(),
        use_container_width=True,
        hide_index=True,
    )
    
    st.markdown(
        """
    For **Accuracy, Precision, Recall, F1 Score, and AUC**,
    higher values are better.
    
    For the **Brier Score**, lower values are better.
    """
    )
    
    st.subheader("Month-by-Month Test Predictions")
    if test_predictions is not None and not test_predictions.empty:
        total_test = len(test_predictions)
        correct_test = (
            int(pd.to_numeric(test_predictions["is_correct"], errors="coerce").fillna(0).sum())
            if "is_correct" in test_predictions.columns
            else "N/A"
        )
        test_col1, test_col2 = st.columns(2)
        test_col1.metric("Test Set Months", total_test)
        test_col2.metric("Correct Predictions", correct_test)
        test_display = round_display_columns(test_predictions, ["predicted_probability"], 4)
        st.dataframe(test_display, use_container_width=True, hide_index=True)

        if {"Date", "CHSO_cases", "predicted_probability"}.issubset(test_predictions.columns):
            fig_test_prob = px.line(
                test_predictions,
                x="Date",
                y="predicted_probability",
                markers=True,
                title="Test Set Predicted Outbreak Probability",
            )
            fig_test_prob.update_yaxes(range=[0, 1])
            st.plotly_chart(fig_test_prob, use_container_width=True)
    else:
        st.warning("test_predictions.csv is unavailable.")


# =============================================================================
# TAB 4: FEATURE TRANSPARENCY
# =============================================================================
with tab4:
    st.header("Feature Transparency")

    st.subheader("Primary Contributing Features")
    if feature_importance is not None and not feature_importance.empty:
        feature_importance = feature_importance.copy()
        if "importance_mean" not in feature_importance.columns and "importance" in feature_importance.columns:
            feature_importance = feature_importance.rename(columns={"importance": "importance_mean"})
        st.dataframe(
            round_display_columns(feature_importance, ["importance_mean", "importance_std"], 6),
            use_container_width=True,
            hide_index=True,
        )
        if {"feature", "importance_mean"}.issubset(feature_importance.columns):
            importance_plot = feature_importance.copy()
            importance_plot["importance_mean"] = pd.to_numeric(importance_plot["importance_mean"], errors="coerce")
            importance_plot = importance_plot.dropna(subset=["importance_mean"])
            fig_importance = px.bar(
                importance_plot.sort_values("importance_mean", ascending=True).tail(15),
                x="importance_mean",
                y="feature",
                orientation="h",
                title="Top 15 Primary Contributing Features",
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("feature_importance.csv is unavailable.")

    st.subheader("Sensitivity Analysis")

    if feature_sensitivity is not None and not feature_sensitivity.empty:
        sens_numeric_cols = [
            "base_avg_outbreak_probability",
            "new_avg_outbreak_probability",
            "delta_probability",
            "percent_change",
        ]
    
        st.dataframe(
            round_display_columns(
                feature_sensitivity,
                sens_numeric_cols,
                6,
            ),
            use_container_width=True,
            hide_index=True,
        )
    
        if {"feature", "delta_probability"}.issubset(
            feature_sensitivity.columns
        ):
            sens_plot = round_display_columns(
                feature_sensitivity,
                ["delta_probability"],
                6,
            ).copy()
    
            sens_plot["color"] = np.where(
                sens_plot["delta_probability"] < 0,
                "Negative",
                "Positive",
            )
    
            fig_sens = px.bar(
                sens_plot,
                x="delta_probability",
                y="feature",
                orientation="h",
                color="color",
                color_discrete_map={
                    "Positive": "#66BB6A",
                    "Negative": "#E73F1E",
                },
                text="delta_probability",
                title="Sensitivity Analysis: Effect of +10% Change in Climate Variables",
            )
    
            fig_sens.update_traces(
                texttemplate="%{text:.4f}",
                textposition="outside",
                cliponaxis=False,
            )
    
            fig_sens.update_layout(
                showlegend=False,
                xaxis_title="Change in Average Outbreak Probability",
                yaxis_title="Climate Variable",
            )
    
            fig_sens.add_vline(
                x=0,
                line_width=1,
                line_color="black",
            )
    
            st.plotly_chart(
                fig_sens,
                use_container_width=True,
            )
    
    else:
        st.warning("feature_sensitivity.csv is unavailable.")
    
    st.info(
        "Feature importance and sensitivity analysis explain model behavior. "
        "These outputs support interpretability but do not prove direct "
        "biological causation."
    )

# =============================================================================
# TAB 5: FORECAST & LIVE PREDICTION
# =============================================================================
with tab5:
    st.header("Forecast & Live Prediction")

    st.subheader("Five-Year Forecast")
    if forecast is not None and not forecast.empty:
        forecast_display = round_display_columns(
            forecast.head(30),
            [
                "rainfall",
                "relative_humidity",
                "temp_mid",
                "predicted_outbreak_probability",
                "predicted_outbreak_probability_lower",
                "predicted_outbreak_probability_upper",
                "predicted_city_cases_proxy",
            ],
            4,
        )
        st.dataframe(forecast_display, use_container_width=True, hide_index=True)

        if {"Date", "predicted_outbreak_probability"}.issubset(forecast.columns):
            fig_forecast = px.line(
                forecast,
                x="Date",
                y="predicted_outbreak_probability",
                markers=True,
                title="5-Year Forecasted Outbreak Probability",
            )
            fig_forecast.update_yaxes(range=[0, 1])
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
        st.warning("forecast_5yr.csv is unavailable.")

    st.subheader("Three Barangays with the Highest Predicted Risk for Forecast Months")
    if forecast_top3_barangays is not None and not forecast_top3_barangays.empty:
        display_forecast_top3 = round_display_columns(
            forecast_top3_barangays,
            [
                "overall_share",
                "recent_share",
                "seasonal_share",
                "risk_score_raw",
                "risk_score",
                "predicted_outbreak_probability",
                "predicted_city_cases_proxy",
                "predicted_barangay_cases_proxy",
            ],
            4,
        )
        st.dataframe(display_forecast_top3, use_container_width=True, hide_index=True)

        if "Date" in forecast_top3_barangays.columns and {"Barangay", "predicted_barangay_cases_proxy"}.issubset(
            forecast_top3_barangays.columns
        ):
            month_options = forecast_top3_barangays["Date"].dropna().dt.strftime("%Y-%m").unique().tolist()
            if not month_options:
                month_options = forecast_top3_barangays["Date"].dropna().astype(str).unique().tolist()
            if month_options:
                selected_month = st.selectbox("Select forecast month for barangay ranking", month_options)
                if np.issubdtype(forecast_top3_barangays["Date"].dtype, np.datetime64):
                    selected_barangay_forecast = forecast_top3_barangays[
                        forecast_top3_barangays["Date"].dt.strftime("%Y-%m") == selected_month
                    ].copy()
                else:
                    selected_barangay_forecast = forecast_top3_barangays[
                        forecast_top3_barangays["Date"].astype(str) == selected_month
                    ].copy()

                fig_barangay_forecast = px.bar(
                    round_display_columns(selected_barangay_forecast, ["predicted_barangay_cases_proxy"], 2),
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
        "Select a target year-month, enter the target month's climate values and the previous three months of dengue cases, then click Predict. The dashboard prepares lag, rolling, and seasonal model inputs automatically."
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
        temp_default = get_profile_value(selected_month_num, "temp_mid", month_profile, monthly, 0.0)

        if target_forecast_row is not None:
            rainfall_default = safe_to_float(target_forecast_row.get("rainfall", rainfall_default), rainfall_default)
            humidity_default = safe_to_float(target_forecast_row.get("relative_humidity", humidity_default), humidity_default)
            temp_default = safe_to_float(target_forecast_row.get("temp_mid", temp_default), temp_default)

        rain_hist_min, rain_hist_max = get_reasonable_range(monthly, "rainfall", 0.0, 1000.0)
        rh_hist_min, rh_hist_max = get_reasonable_range(monthly, "relative_humidity", 60.0, 100.0)
        temp_hist_min, temp_hist_max = get_reasonable_range(monthly, "temp_mid", 10.0, 35.0)
        cases_hist_min, cases_hist_max = get_reasonable_range(monthly, "CHSO_cases", 0.0, 3000.0)

        rain_min = min(0.0, rain_hist_min, rainfall_default)
        rain_max = max(1000.0, rain_hist_max, rainfall_default + 1.0)
        rh_min = min(60.0, rh_hist_min, humidity_default)
        rh_max = max(100.0, rh_hist_max, humidity_default + 1.0)
        temp_min = min(10.0, temp_hist_min, temp_default)
        temp_max = max(35.0, temp_hist_max, temp_default + 1.0)

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
                value=float(round(humidity_default, 2)),
                step=0.1,
            )
        with climate_col3:
            temp_now = st.slider(
                "Current Temperature (°C)",
                min_value=float(round(temp_min, 2)),
                max_value=float(round(temp_max, 2)),
                value=float(round(temp_default, 2)),
                step=0.1,
            )

        st.markdown("### Recent Dengue Case History")
        if target_forecast_row is not None:
            default_cases_lag_1 = safe_to_float(target_forecast_row.get("cases_lag_1", 0.0), 0.0)
            default_cases_lag_2 = safe_to_float(target_forecast_row.get("cases_lag_2", default_cases_lag_1), default_cases_lag_1)
            default_cases_lag_3 = safe_to_float(target_forecast_row.get("cases_lag_3", default_cases_lag_2), default_cases_lag_2)
        else:
            cases_series = (
                pd.to_numeric(monthly["CHSO_cases"], errors="coerce").dropna()
                if "CHSO_cases" in monthly.columns
                else pd.Series(dtype=float)
            )
            default_cases_lag_1 = float(cases_series.iloc[-1]) if len(cases_series) >= 1 else 0.0
            default_cases_lag_2 = float(cases_series.iloc[-2]) if len(cases_series) >= 2 else default_cases_lag_1
            default_cases_lag_3 = float(cases_series.iloc[-3]) if len(cases_series) >= 3 else default_cases_lag_2

        max_cases_slider = int(max(cases_hist_max, default_cases_lag_1, default_cases_lag_2, default_cases_lag_3, 1))
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
            st.dataframe(pd.DataFrame([input_values]), use_container_width=True, hide_index=True)

        missing_features = [feature for feature in feature_columns if feature not in auto_feature_values]
        if missing_features:
            st.warning(f"Some model features were not generated and were filled with 0: {missing_features}")

        if st.button("Predict", type="primary"):
            input_df = pd.DataFrame([input_values], columns=feature_columns)
            input_df = input_df.apply(pd.to_numeric, errors="coerce").fillna(0.0)

            pred = int(model.predict(input_df)[0])
            if hasattr(model, "predict_proba"):
                prob = float(model.predict_proba(input_df)[0][1])
            else:
                prob = np.nan

            result_col1, result_col2 = st.columns(2)

            if pred == 1:
                result_col1.error("Predicted Class: Outbreak")
            else:
                result_col1.success("Predicted Class: Non-outbreak")
            
            result_col2.info(
                f"Predicted Outbreak Probability: {prob:.4f}"
                if not pd.isna(prob)
                else "Probability not available"
            )

            st.caption("0 = Non-outbreak month; 1 = Outbreak month. Probability is the model's estimated likelihood of the outbreak class.")

            st.subheader("Barangays with the Highest Predicted Risk")
            if barangay_risk_profile is not None and not barangay_risk_profile.empty:
                barangay_live = barangay_risk_profile.copy()
                for col in ["overall_share", "recent_share", "seasonal_share"]:
                    if col not in barangay_live.columns:
                        barangay_live[col] = 0.0

                if "Month" in barangay_live.columns:
                    seasonal_subset = barangay_live[
                        pd.to_numeric(barangay_live["Month"], errors="coerce") == selected_month_num
                    ].copy()
                    if not seasonal_subset.empty:
                        barangay_live = seasonal_subset

                barangay_live["risk_score_raw"] = (
                    0.50 * pd.to_numeric(barangay_live["seasonal_share"], errors="coerce").fillna(0)
                    + 0.30 * pd.to_numeric(barangay_live["recent_share"], errors="coerce").fillna(0)
                    + 0.20 * pd.to_numeric(barangay_live["overall_share"], errors="coerce").fillna(0)
                )
                total_score = float(barangay_live["risk_score_raw"].sum())
                if total_score > 0:
                    barangay_live["risk_score"] = barangay_live["risk_score_raw"] / total_score
                else:
                    barangay_live["risk_score"] = 0.0

                city_cases_proxy = float(input_values.get("cases_roll3_mean", 0.0)) * (1 + (0 if pd.isna(prob) else prob))
                barangay_live["predicted_city_cases_proxy"] = city_cases_proxy
                barangay_live["predicted_barangay_cases_proxy"] = barangay_live["risk_score"] * city_cases_proxy
                barangay_live["predicted_barangay_label"] = "Higher Risk"

                keep_cols = [
                    col
                    for col in [
                        "Barangay",
                        "overall_share",
                        "recent_share",
                        "seasonal_share",
                        "risk_score_raw",
                        "risk_score",
                        "predicted_city_cases_proxy",
                        "predicted_barangay_cases_proxy",
                        "predicted_barangay_label",
                    ]
                    if col in barangay_live.columns
                ]

                barangay_live_high_risk = barangay_live.sort_values(
                    "predicted_barangay_cases_proxy", ascending=False
                ).head(3)
                st.dataframe(
                    round_display_columns(
                        barangay_live_high_risk[keep_cols],
                        [
                            "overall_share",
                            "recent_share",
                            "seasonal_share",
                            "risk_score_raw",
                            "risk_score",
                            "predicted_city_cases_proxy",
                            "predicted_barangay_cases_proxy",
                        ],
                        4,
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                if {"Barangay", "predicted_barangay_cases_proxy"}.issubset(barangay_live_high_risk.columns):
                    fig_live_barangay = px.bar(
                        round_display_columns(barangay_live_high_risk, ["predicted_barangay_cases_proxy"], 2),
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
