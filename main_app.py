import json
import warnings
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

# =============================================================================
# CONFIGURATION
# =============================================================================
APP_TITLE = "Baguio City Dengue Forecast Dashboard"
APP_CAPTION = "Interactive dashboard for dengue outbreak forecasting using Baguio City climate and epidemiological artifacts."

# The app first looks inside artifacts/, then beside main_app.py. The /mnt/data path
# is only a fallback for local testing in this ChatGPT sandbox and will be ignored
# on normal Streamlit/GitHub deployment if it does not exist.
ARTIFACT_DIR_CANDIDATES = [Path("artifacts"), Path("."), Path("/mnt/data")]

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

DATE_ARTIFACTS = {
    "monthly",
    "forecast",
    "barangay_monthly",
    "top_barangay_monthly",
    "test_predictions",
    "forecast_barangay_ranking",
    "forecast_top3_barangays",
}

MONTH_NAMES = {
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

st.markdown(
    """
<style>
.block-container {padding-top: 1.25rem; padding-bottom: 2rem;}
[data-testid="stMetricValue"] {font-size: 1.8rem;}
.small-note {font-size: 0.88rem; color: #666;}
.warning-note {font-size: 0.9rem; color: #9a6700;}
</style>
""",
    unsafe_allow_html=True,
)


# =============================================================================
# LOADING HELPERS
# =============================================================================
def first_existing_path(file_names):
    for folder in ARTIFACT_DIR_CANDIDATES:
        for file_name in file_names:
            path = folder / file_name
            if path.exists() and path.is_file():
                return path
    return None


def safe_read_csv(file_names):
    path = first_existing_path(file_names)
    if path is None:
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:
        st.sidebar.warning(f"Could not read {path.name}: {exc}")
        return None


def safe_read_json(file_name="meta.json"):
    path = first_existing_path([file_name])
    if path is None:
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        st.sidebar.warning(f"Could not read {path.name}: {exc}")
        return None


def safe_load_model(file_name="best_model.joblib"):
    path = first_existing_path([file_name])
    if path is None:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return joblib.load(path)
    except Exception as exc:
        st.sidebar.warning(f"Could not load {path.name}: {exc}")
        return None


@st.cache_data(show_spinner=False)
def load_artifacts_from_disk():
    loaded = {key: safe_read_csv(names) for key, names in ARTIFACT_SPECS.items()}
    loaded["meta"] = safe_read_json("meta.json")
    return loaded


@st.cache_resource(show_spinner=False)
def load_model_from_disk():
    return safe_load_model("best_model.joblib")


# =============================================================================
# CLEANING AND NORMALIZATION HELPERS
# =============================================================================
def month_name_from_number(month_num):
    try:
        return MONTH_NAMES.get(int(month_num), str(month_num))
    except Exception:
        return str(month_num)


def parse_date_column(df):
    if df is None or df.empty or "Date" not in df.columns:
        return df
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


def numeric_series(df, col):
    if df is None or col not in df.columns:
        return pd.Series(dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def safe_metric_value(value, decimals=2):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


def round_display_columns(df, columns, decimals=2):
    if df is None:
        return None
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
    return df


def rename_existing_columns(df, rename_map):
    if df is None:
        return None
    return df.rename(columns={old: new for old, new in rename_map.items() if old in df.columns})


def display_clean_barangay_columns(df):
    if df is None:
        return None
    rename_map = {
        "Top_Barangay": "Barangay",
        "Top_Barangay_Cases": "Barangay_cases",
        "rank_within_year": "Rank Within Year",
        "rank": "Rank",
        "Barangay_cases": "Dengue Cases",
        "predicted_barangay_cases_proxy": "Predicted Barangay Cases Proxy",
        "predicted_city_cases_proxy": "Predicted City Cases Proxy",
        "predicted_outbreak_probability": "Predicted Outbreak Probability",
    }
    return rename_existing_columns(df.copy(), rename_map)


def normalize_model_comparison(model_df, auc_table):
    if model_df is None or model_df.empty:
        return model_df

    df = model_df.copy()
    df = rename_existing_columns(
        df,
        {
            "Model": "model",
            "F1 Score": "f1_score",
            "F1": "f1_score",
            "Precision": "precision",
            "Recall": "recall",
            "Accuracy": "accuracy",
            "Reliability (Brier)": "brier",
            "Brier": "brier",
            "AUC (Supplementary)": "auc",
            "AUC": "auc",
        },
    )

    if "model" not in df.columns:
        return df

    if auc_table is not None and not auc_table.empty:
        auc_tmp = auc_table.copy()
        auc_tmp = rename_existing_columns(auc_tmp, {"Model": "model", "AUC": "auc"})
        if {"model", "auc"}.issubset(auc_tmp.columns):
            if "auc" in df.columns:
                df = df.drop(columns=["auc"])
            df = df.merge(auc_tmp[["model", "auc"]], on="model", how="left")

    metric_cols = [c for c in ["accuracy", "precision", "recall", "f1_score", "auc", "brier"] if c in df.columns]
    for col in metric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Prefer the strongest balanced metric first when showing/ranking models.
    sort_cols = [c for c in ["f1_score", "recall", "precision", "auc"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=[False] * len(sort_cols), na_position="last")
    return df


def normalize_feature_sensitivity(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df = rename_existing_columns(
        df,
        {
            "new_avg_outbreak_probability (10% increase)": "new_avg_outbreak_probability",
            "change_in_probability": "delta_probability",
            "Percent Change": "percent_change",
        },
    )
    return df


def normalize_feature_importance(df):
    if df is None or df.empty:
        return df
    df = df.copy()
    df = rename_existing_columns(df, {"importance": "importance_mean", "Feature": "feature"})
    if "importance_mean" in df.columns:
        df["importance_mean"] = pd.to_numeric(df["importance_mean"], errors="coerce")
        df = df.sort_values("importance_mean", ascending=False)
    return df


def complete_month_profile(month_profile_df, monthly_df):
    if monthly_df is None or monthly_df.empty or "Month" not in monthly_df.columns:
        return month_profile_df

    needed_cols = [c for c in ["CHSO_cases", "rainfall", "relative_humidity", "temp_mid"] if c in monthly_df.columns]
    if not needed_cols:
        return month_profile_df

    generated = monthly_df.groupby("Month", as_index=False)[needed_cols].mean(numeric_only=True)
    generated["MonthName"] = generated["Month"].apply(month_name_from_number)

    if month_profile_df is None or month_profile_df.empty or "Month" not in month_profile_df.columns:
        return generated

    fixed = month_profile_df.copy()
    for col in ["CHSO_cases", "rainfall", "relative_humidity", "temp_mid"]:
        if col not in fixed.columns and col in generated.columns:
            fixed = fixed.merge(generated[["Month", col]], on="Month", how="left")
    if "MonthName" not in fixed.columns:
        fixed["MonthName"] = fixed["Month"].apply(month_name_from_number)
    return fixed


def build_climate_case_correlation(monthly_df):
    if monthly_df is None or monthly_df.empty or "CHSO_cases" not in monthly_df.columns:
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


def get_profile_value(month_num, col_name, month_profile_df, fallback_df=None, default=0.0):
    if month_profile_df is not None and not month_profile_df.empty and "Month" in month_profile_df.columns:
        month_series = pd.to_numeric(month_profile_df["Month"], errors="coerce")
        subset = month_profile_df[month_series == int(month_num)]
        if not subset.empty and col_name in subset.columns:
            value = pd.to_numeric(pd.Series([subset.iloc[0][col_name]]), errors="coerce").iloc[0]
            if pd.notna(value):
                return float(value)

    if fallback_df is not None and col_name in fallback_df.columns:
        values = pd.to_numeric(fallback_df[col_name], errors="coerce").dropna()
        if len(values) > 0:
            return float(values.mean())
    return float(default)


def combined_numeric_values(col_name, *dfs):
    pieces = []
    for df in dfs:
        if df is not None and col_name in df.columns:
            pieces.append(pd.to_numeric(df[col_name], errors="coerce"))
    if not pieces:
        return pd.Series(dtype=float)
    return pd.concat(pieces, ignore_index=True).dropna()


def reasonable_number_input_bounds(col_name, default_min, default_max, *dfs, pad_ratio=0.10):
    values = combined_numeric_values(col_name, *dfs)
    if values.empty:
        return float(default_min), float(default_max)
    vmin = float(values.min())
    vmax = float(values.max())
    if vmin == vmax:
        vmin -= 1.0
        vmax += 1.0
    pad = (vmax - vmin) * pad_ratio
    return float(min(default_min, vmin - pad)), float(max(default_max, vmax + pad))


def get_forecast_row(forecast_df, year_num, month_num):
    if forecast_df is None or forecast_df.empty or not {"Year", "Month"}.issubset(forecast_df.columns):
        return None
    year_series = pd.to_numeric(forecast_df["Year"], errors="coerce")
    month_series = pd.to_numeric(forecast_df["Month"], errors="coerce")
    subset = forecast_df[(year_series == int(year_num)) & (month_series == int(month_num))]
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
        def row_value(col, fallback=0.0):
            value = pd.to_numeric(pd.Series([forecast_row.get(col, fallback)]), errors="coerce").iloc[0]
            return float(value) if pd.notna(value) else float(fallback)

        rainfall_lag_1 = row_value("rainfall_lag_1")
        rainfall_lag_2 = row_value("rainfall_lag_2")
        rainfall_lag_3 = row_value("rainfall_lag_3")
        rh_lag_1 = row_value("relative_humidity_lag_1")
        rh_lag_2 = row_value("relative_humidity_lag_2")
        rh_lag_3 = row_value("relative_humidity_lag_3")
        temp_lag_1 = row_value("temp_mid_lag_1")
        temp_lag_2 = row_value("temp_mid_lag_2")
        temp_lag_3 = row_value("temp_mid_lag_3")
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

    cases_lag_values = [float(cases_lag_1), float(cases_lag_2), float(cases_lag_3)]

    return {
        "rainfall": float(rainfall_now),
        "relative_humidity": float(humidity_now),
        "temp_mid": float(temp_now),
        "cases_lag_1": float(cases_lag_1),
        "cases_lag_2": float(cases_lag_2),
        "cases_lag_3": float(cases_lag_3),
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


def outbreak_label_from_binary(value):
    try:
        return "Outbreak" if int(value) == 1 else "Non-outbreak"
    except Exception:
        return "Unknown"


def make_probability_label(prob):
    if pd.isna(prob):
        return "N/A"
    return f"{float(prob):.4f}"


def get_feature_columns(meta, model):
    if model is not None and hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)
    if meta and isinstance(meta.get("feature_columns"), list):
        return list(meta["feature_columns"])
    return DEFAULT_FEATURE_COLS


def safe_predict(model, input_df):
    pred = int(model.predict(input_df)[0])
    prob = np.nan
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        if len(probabilities) > 1:
            prob = float(probabilities[1])
    return pred, prob


def build_live_barangay_priority(barangay_risk_profile, selected_month_num, prob, city_cases_proxy):
    if barangay_risk_profile is None or barangay_risk_profile.empty:
        return None
    df = barangay_risk_profile.copy()
    for col in ["overall_share", "recent_share", "seasonal_share"]:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    if "Month" in df.columns:
        month_subset = df[pd.to_numeric(df["Month"], errors="coerce") == int(selected_month_num)].copy()
        if not month_subset.empty:
            df = month_subset

    df["risk_score_raw"] = (
        0.50 * df["seasonal_share"]
        + 0.30 * df["recent_share"]
        + 0.20 * df["overall_share"]
    )
    total_score = float(df["risk_score_raw"].sum())
    df["risk_score"] = df["risk_score_raw"] / total_score if total_score > 0 else 0.0
    df["predicted_outbreak_probability"] = np.nan if pd.isna(prob) else float(prob)
    df["predicted_city_cases_proxy"] = float(city_cases_proxy)
    df["predicted_barangay_cases_proxy"] = df["risk_score"] * float(city_cases_proxy)
    df["predicted_barangay_label"] = "Higher Risk"
    return df.sort_values("predicted_barangay_cases_proxy", ascending=False).head(3)


# =============================================================================
# LOAD ARTIFACTS
# =============================================================================
artifacts = load_artifacts_from_disk()
model = load_model_from_disk()

# Sidebar manual upload fallback.
st.sidebar.header("Dashboard Files")
st.sidebar.write("Use the `artifacts/` folder exported from Google Colab, or upload replacement files below.")

with st.sidebar.expander("Manual file upload", expanded=False):
    uploaded_csvs = {}
    upload_labels = {
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
        "forecast_barangay_ranking": "forecast_barangay_ranking.csv",
        "forecast_top3_barangays": "forecast_top3_barangays.csv",
        "barangay_risk_profile": "barangay_risk_profile.csv",
    }
    for key, label in upload_labels.items():
        uploaded_csvs[key] = st.file_uploader(label, type=["csv"], key=f"upload_{key}")
    uploaded_meta = st.file_uploader("meta.json", type=["json"], key="upload_meta")
    uploaded_model = st.file_uploader("best_model.joblib", type=["joblib", "pkl"], key="upload_model")

for key, uploaded_file in uploaded_csvs.items():
    if uploaded_file is not None:
        try:
            artifacts[key] = pd.read_csv(uploaded_file)
        except Exception as exc:
            st.sidebar.error(f"Could not read uploaded {upload_labels[key]}: {exc}")

if uploaded_meta is not None:
    try:
        artifacts["meta"] = json.load(uploaded_meta)
    except Exception as exc:
        st.sidebar.error(f"Could not read uploaded meta.json: {exc}")

if uploaded_model is not None:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = joblib.load(uploaded_model)
    except Exception as exc:
        st.sidebar.error(f"Could not load uploaded model: {exc}")

for key in DATE_ARTIFACTS:
    artifacts[key] = parse_date_column(artifacts.get(key))

meta = artifacts.get("meta")
monthly = artifacts.get("monthly")
model_comparison = artifacts.get("model_comparison")
auc_df = artifacts.get("auc_df")
feature_importance = normalize_feature_importance(artifacts.get("feature_importance"))
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

if monthly is None or monthly.empty:
    st.error("monthly_modeling_dataset.csv is required. Export it from the revised Colab workflow, place it in the artifacts folder, or upload it in the sidebar.")
    st.stop()

month_profile = complete_month_profile(month_profile, monthly)
model_comparison = normalize_model_comparison(model_comparison, auc_df)
if climate_case_correlation is None or climate_case_correlation.empty:
    climate_case_correlation = build_climate_case_correlation(monthly)

feature_columns = get_feature_columns(meta, model)

# Sidebar metadata summary.
if meta:
    st.sidebar.success(f"Best Model: {meta.get('best_model', 'Unknown')}")
    threshold_value = meta.get("outbreak_threshold_cases", meta.get("threshold", None))
    if threshold_value is not None:
        st.sidebar.info(f"Outbreak Threshold: {safe_metric_value(threshold_value, 2)} cases")
    st.sidebar.caption(f"Train/Test: {meta.get('train_months', 'N/A')} / {meta.get('test_months', 'N/A')} months")
    st.sidebar.caption(f"Forecast Period: {meta.get('forecast_period', 'N/A')}")
else:
    st.sidebar.warning("meta.json was not found. The app will use default feature columns.")

missing_features = [col for col in feature_columns if col not in monthly.columns and (forecast is None or col not in forecast.columns)]
if missing_features:
    st.sidebar.warning("Some model features were not found in the monthly/forecast artifacts: " + ", ".join(missing_features))


# =============================================================================
# PAGE HEADER
# =============================================================================
st.title(APP_TITLE)
st.caption(APP_CAPTION)

if meta:
    with st.expander("Study and Model Metadata", expanded=False):
        st.write(f"**Problem Definition:** {meta.get('problem_definition', 'Monthly dengue outbreak classification')}")
        st.write(f"**Outbreak Definition:** {meta.get('outbreak_definition', 'CHSO monthly cases greater than or equal to the selected threshold.')}")
        meta_cols = st.columns(4)
        meta_cols[0].metric("Best Model", meta.get("best_model", "N/A"))
        meta_cols[1].metric("Train Months", meta.get("train_months", "N/A"))
        meta_cols[2].metric("Test Months", meta.get("test_months", "N/A"))
        meta_cols[3].metric("Forecast Months", meta.get("forecast_months", "N/A"))

# Data quality note: temp_mid should normally be a temperature-like input, but the app
# intentionally follows the exported artifact range exactly to stay consistent with the
# trained joblib model.
temp_values = combined_numeric_values("temp_mid", monthly, forecast)
if not temp_values.empty and float(temp_values.max()) > 60:
    st.warning(
        "The exported `temp_mid` values exceed typical Celsius temperature ranges. "
        "The dashboard still uses these exact artifact values so predictions remain consistent with the trained model. "
        "Check the Colab climate preprocessing cell if this was not intended."
    )


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

with tab1:
    st.header("Historical Dengue Overview")

    total_months = len(monthly)
    total_cases = int(numeric_series(monthly, "CHSO_cases").fillna(0).sum()) if "CHSO_cases" in monthly.columns else 0
    avg_cases = numeric_series(monthly, "CHSO_cases").mean() if "CHSO_cases" in monthly.columns else np.nan
    outbreak_months = int(numeric_series(monthly, "outbreak").fillna(0).sum()) if "outbreak" in monthly.columns else "N/A"

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Months", total_months)
    col2.metric("Total CHSO Cases", f"{total_cases:,}")
    col3.metric("Average Monthly Cases", safe_metric_value(avg_cases))
    col4.metric("Outbreak Months", outbreak_months)

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
            )
        else:
            fig_line = px.line(
                monthly,
                x="Date",
                y="CHSO_cases",
                markers=True,
                title="Monthly Dengue Cases in Baguio City (CHSO)",
            )
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.info("Monthly line chart requires Date and CHSO_cases columns.")

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

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.subheader("Climate-Case Correlation")
        if climate_case_correlation is not None and not climate_case_correlation.empty:
            corr_display = round_display_columns(climate_case_correlation, ["pearson_corr_with_CHSO_cases"], 4)
            st.dataframe(corr_display, use_container_width=True)
            if {"feature", "pearson_corr_with_CHSO_cases"}.issubset(climate_case_correlation.columns):
                fig_corr = px.bar(
                    corr_display,
                    x="feature",
                    y="pearson_corr_with_CHSO_cases",
                    text="pearson_corr_with_CHSO_cases",
                    title="Pearson Correlation with CHSO Cases",
                )
                fig_corr.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.warning("Climate-case correlation table is unavailable.")

    with chart_col2:
        st.subheader("Average Monthly Profile")
        if month_profile is not None and not month_profile.empty:
            display_cols = [c for c in ["Month", "MonthName", "CHSO_cases", "rainfall", "relative_humidity", "temp_mid"] if c in month_profile.columns]
            numeric_cols = [c for c in display_cols if c != "MonthName"]
            st.dataframe(round_display_columns(month_profile[display_cols], numeric_cols, 2), use_container_width=True)
            if {"MonthName", "CHSO_cases"}.issubset(month_profile.columns):
                fig_month = px.bar(
                    round_display_columns(month_profile, ["CHSO_cases"], 2),
                    x="MonthName",
                    y="CHSO_cases",
                    text="CHSO_cases",
                    title="Average CHSO Cases by Month",
                )
                fig_month.update_traces(texttemplate="%{text:.2f}", textposition="outside")
                st.plotly_chart(fig_month, use_container_width=True)
        else:
            st.warning("month_profile.csv is unavailable.")

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
        fig_climate_profile = px.bar(
            round_display_columns(climate_long, ["Average Value"], 2),
            x="Climate Variable",
            y="Average Value",
            color="Outbreak Status",
            barmode="group",
            text="Average Value",
            title="Climate Profile of Outbreak vs Non-outbreak Months",
        )
        fig_climate_profile.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        st.plotly_chart(fig_climate_profile, use_container_width=True)

with tab2:
    st.header("Barangay Analytics")

    st.subheader("Barangay with the Highest Monthly Dengue Cases")
    if top_barangay_monthly is not None and not top_barangay_monthly.empty:
        st.dataframe(display_clean_barangay_columns(top_barangay_monthly), use_container_width=True)
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
            st.dataframe(display_clean_barangay_columns(top3_barangays_yearly), use_container_width=True)
            if {"Year", "Barangay", "Barangay_cases"}.issubset(top3_barangays_yearly.columns):
                fig_tree = px.treemap(
                    top3_barangays_yearly,
                    path=["Year", "Barangay"],
                    values="Barangay_cases",
                    color="Barangay_cases",
                    title="Three Barangays with the Highest Dengue Cases per Year",
                )
                fig_tree.update_layout(height=650)
                st.plotly_chart(fig_tree, use_container_width=True)
        else:
            st.warning("top3_barangays_yearly.csv is unavailable.")
    else:
        if top3_barangays_overall is not None and not top3_barangays_overall.empty:
            st.dataframe(display_clean_barangay_columns(top3_barangays_overall), use_container_width=True)
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
        if "Barangay" in barangay_monthly.columns:
            barangay_options = ["All"] + sorted(barangay_monthly["Barangay"].dropna().astype(str).unique().tolist())
            selected_barangay = st.selectbox("Filter barangay", barangay_options)
            barangay_view = barangay_monthly.copy()
            if selected_barangay != "All":
                barangay_view = barangay_view[barangay_view["Barangay"].astype(str) == selected_barangay]
        else:
            barangay_view = barangay_monthly.copy()
        st.dataframe(display_clean_barangay_columns(barangay_view), use_container_width=True)
    else:
        st.warning("barangay_monthly.csv is unavailable.")

with tab3:
    st.header("Model Results")

    if meta:
        st.success(f"Selected Model: {meta.get('best_model', 'Unknown')}")

    if model_comparison is not None and not model_comparison.empty:
        display_cols = [c for c in ["model", "f1_score", "precision", "recall", "auc", "brier", "accuracy"] if c in model_comparison.columns]
        metric_cols = [c for c in display_cols if c != "model"]
        st.dataframe(round_display_columns(model_comparison[display_cols], metric_cols, 4), use_container_width=True)

        plot_metrics = [c for c in ["f1_score", "precision", "recall", "auc"] if c in model_comparison.columns]
        if "model" in model_comparison.columns and plot_metrics:
            st.subheader("Model Comparison by Metric")
            results_long = model_comparison.melt(
                id_vars="model",
                value_vars=plot_metrics,
                var_name="Metric",
                value_name="Score",
            )
            fig_model = px.bar(
                round_display_columns(results_long, ["Score"], 4),
                x="model",
                y="Score",
                color="Metric",
                barmode="group",
                text="Score",
                title="Model Comparison by Classification Metric",
            )
            fig_model.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            fig_model.update_yaxes(range=[0, 1.15])
            st.plotly_chart(fig_model, use_container_width=True)

        if "brier" in model_comparison.columns and "model" in model_comparison.columns:
            fig_brier = px.bar(
                round_display_columns(model_comparison, ["brier"], 4),
                x="model",
                y="brier",
                text="brier",
                title="Reliability by Brier Score Lower is Better",
            )
            fig_brier.update_traces(texttemplate="%{text:.4f}", textposition="outside")
            st.plotly_chart(fig_brier, use_container_width=True)
    else:
        st.warning("model_comparison.csv is unavailable.")

    st.subheader("How to Read the Metrics")
    st.markdown(
        """
**Precision** measures how often predicted outbreak months are actual outbreaks. **Recall/Sensitivity** measures how many actual outbreak months are detected. **F1 score** balances precision and recall. **AUC** summarizes separation between outbreak and non-outbreak months across probability thresholds. **Brier score** measures probability reliability, where lower values indicate better calibrated probabilities.
"""
    )

    st.subheader("Month-by-Month Test Predictions")
    if test_predictions is not None and not test_predictions.empty:
        total_test = len(test_predictions)
        correct_test = int(numeric_series(test_predictions, "is_correct").fillna(0).sum()) if "is_correct" in test_predictions.columns else "N/A"
        c1, c2, c3 = st.columns(3)
        c1.metric("Test Set Months", total_test)
        c2.metric("Correct Predictions", correct_test)
        if isinstance(correct_test, int) and total_test > 0:
            c3.metric("Test Accuracy", f"{correct_test / total_test:.2%}")
        else:
            c3.metric("Test Accuracy", "N/A")

        st.dataframe(test_predictions, use_container_width=True)

        if {"Date", "outbreak", "predicted_probability"}.issubset(test_predictions.columns):
            test_plot = test_predictions.copy()
            test_plot["Date"] = pd.to_datetime(test_plot["Date"], errors="coerce")
            fig_test = px.line(
                test_plot,
                x="Date",
                y="predicted_probability",
                markers=True,
                title="Test Set Predicted Outbreak Probability",
                hover_data=[c for c in ["actual_label", "predicted_label", "CHSO_cases", "is_correct"] if c in test_plot.columns],
            )
            fig_test.add_hline(y=0.5, line_dash="dash", annotation_text="0.50 decision reference")
            st.plotly_chart(fig_test, use_container_width=True)
    else:
        st.warning("test_predictions.csv is unavailable.")

with tab4:
    st.header("Feature Transparency")

    st.subheader("Primary Contributing Features")
    if feature_importance is not None and not feature_importance.empty:
        numeric_cols = [c for c in ["importance_mean", "importance_std"] if c in feature_importance.columns]
        st.dataframe(round_display_columns(feature_importance, numeric_cols, 6), use_container_width=True)
        if {"feature", "importance_mean"}.issubset(feature_importance.columns):
            fig_importance = px.bar(
                feature_importance.sort_values("importance_mean", ascending=True).tail(15),
                x="importance_mean",
                y="feature",
                orientation="h",
                title="Top Model Features by Mean Importance",
            )
            st.plotly_chart(fig_importance, use_container_width=True)
    else:
        st.warning("feature_importance.csv is unavailable.")

    st.subheader("Sensitivity Analysis")
    if feature_sensitivity is not None and not feature_sensitivity.empty:
        numeric_cols = [
            c
            for c in [
                "base_avg_outbreak_probability",
                "new_avg_outbreak_probability",
                "delta_probability",
                "percent_change",
            ]
            if c in feature_sensitivity.columns
        ]
        st.dataframe(round_display_columns(feature_sensitivity, numeric_cols, 6), use_container_width=True)
        if {"feature", "delta_probability"}.issubset(feature_sensitivity.columns):
            fig_sens = px.bar(
                round_display_columns(feature_sensitivity, ["delta_probability"], 6),
                x="feature",
                y="delta_probability",
                text="delta_probability",
                title="Effect of +10% Change in Climate Variables on Outbreak Probability",
            )
            fig_sens.update_traces(texttemplate="%{text:.6f}", textposition="outside")
            st.plotly_chart(fig_sens, use_container_width=True)
    else:
        st.warning("feature_sensitivity.csv is unavailable.")

    st.info("Feature importance and sensitivity analysis explain model behavior. They do not, by themselves, prove direct biological causation.")

with tab5:
    st.header("Forecast & Live Prediction")

    st.subheader("Five-Year Forecast")
    if forecast is not None and not forecast.empty:
        forecast_display_cols = [
            c
            for c in [
                "Date",
                "Year",
                "Month",
                "predicted_outbreak_probability",
                "predicted_outbreak_probability_lower",
                "predicted_outbreak_probability_upper",
                "predicted_outbreak",
                "predicted_label",
                "predicted_city_cases_proxy",
            ]
            if c in forecast.columns
        ]
        forecast_numeric_cols = [c for c in forecast_display_cols if c not in ["Date", "predicted_label"]]
        st.dataframe(round_display_columns(forecast[forecast_display_cols].head(60), forecast_numeric_cols, 4), use_container_width=True)

        high_risk_count = int(numeric_series(forecast, "predicted_outbreak").fillna(0).sum()) if "predicted_outbreak" in forecast.columns else "N/A"
        forecast_cols = st.columns(3)
        forecast_cols[0].metric("Forecast Months", len(forecast))
        forecast_cols[1].metric("Predicted Outbreak Months", high_risk_count)
        if "predicted_outbreak_probability" in forecast.columns:
            forecast_cols[2].metric("Average Probability", safe_metric_value(numeric_series(forecast, "predicted_outbreak_probability").mean(), 4))
        else:
            forecast_cols[2].metric("Average Probability", "N/A")

        if {"Date", "predicted_outbreak_probability"}.issubset(forecast.columns):
            fig_forecast = px.line(
                forecast,
                x="Date",
                y="predicted_outbreak_probability",
                markers=True,
                title="5-Year Forecasted Outbreak Probability",
                hover_data=[c for c in ["predicted_label", "predicted_city_cases_proxy"] if c in forecast.columns],
            )
            if {"predicted_outbreak_probability_lower", "predicted_outbreak_probability_upper"}.issubset(forecast.columns):
                band_df = forecast[["Date", "predicted_outbreak_probability_lower", "predicted_outbreak_probability_upper"]].melt(
                    id_vars="Date",
                    value_vars=["predicted_outbreak_probability_lower", "predicted_outbreak_probability_upper"],
                    var_name="Probability Band",
                    value_name="Probability",
                )
                fig_band = px.line(band_df, x="Date", y="Probability", color="Probability Band", line_dash="Probability Band")
                for trace in fig_band.data:
                    fig_forecast.add_trace(trace)
            fig_forecast.add_hline(y=0.5, line_dash="dash", annotation_text="0.50 decision reference")
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
        top3_numeric_cols = [
            c
            for c in [
                "overall_share",
                "recent_share",
                "seasonal_share",
                "risk_score_raw",
                "risk_score",
                "predicted_outbreak_probability",
                "predicted_city_cases_proxy",
                "predicted_barangay_cases_proxy",
            ]
            if c in forecast_top3_barangays.columns
        ]
        st.dataframe(round_display_columns(forecast_top3_barangays, top3_numeric_cols, 4), use_container_width=True)

        if "Date" in forecast_top3_barangays.columns and {"Barangay", "predicted_barangay_cases_proxy"}.issubset(forecast_top3_barangays.columns):
            month_options = forecast_top3_barangays["Date"].dropna().dt.strftime("%Y-%m").unique().tolist()
            if month_options:
                selected_month_str = st.selectbox("Select forecast month for barangay ranking", month_options)
                selected_date = pd.to_datetime(selected_month_str + "-01")
                selected_barangay_forecast = forecast_top3_barangays[forecast_top3_barangays["Date"].dt.to_period("M") == selected_date.to_period("M")].copy()
                fig_barangay_forecast = px.bar(
                    round_display_columns(selected_barangay_forecast, ["predicted_barangay_cases_proxy"], 2),
                    x="Barangay",
                    y="predicted_barangay_cases_proxy",
                    color="Barangay",
                    text="predicted_barangay_cases_proxy",
                    title=f"Three Barangays with the Highest Predicted Risk - {selected_month_str}",
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
**Rainfall, relative humidity, and temp_mid** should refer to the target month itself. **Cases Last Month**, **Cases 2 Months Ago**, and **Cases 3 Months Ago** refer to the three previous months. The output is a monthly outbreak classification, not a confirmed case count. Barangay values are prioritization proxies based on historical barangay risk shares.
"""
        )
        st.write("Model input order:")
        st.code(", ".join(feature_columns), language="text")

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

        select_col1, select_col2 = st.columns(2)
        with select_col1:
            selected_year_num = st.selectbox("Select Year", year_options, index=0)
        with select_col2:
            selected_month_num = st.selectbox(
                "Select Month",
                list(range(1, 13)),
                format_func=lambda x: f"{x} - {month_name_from_number(x)}",
                index=0,
            )

        target_forecast_row = get_forecast_row(forecast, selected_year_num, selected_month_num)

        rainfall_default = get_profile_value(selected_month_num, "rainfall", month_profile, monthly, 0.0)
        humidity_default = get_profile_value(selected_month_num, "relative_humidity", month_profile, monthly, 0.0)
        temp_default = get_profile_value(selected_month_num, "temp_mid", month_profile, monthly, 0.0)

        if target_forecast_row is not None:
            rainfall_default = float(pd.to_numeric(pd.Series([target_forecast_row.get("rainfall", rainfall_default)]), errors="coerce").fillna(rainfall_default).iloc[0])
            humidity_default = float(pd.to_numeric(pd.Series([target_forecast_row.get("relative_humidity", humidity_default)]), errors="coerce").fillna(humidity_default).iloc[0])
            temp_default = float(pd.to_numeric(pd.Series([target_forecast_row.get("temp_mid", temp_default)]), errors="coerce").fillna(temp_default).iloc[0])

        rainfall_min, rainfall_max = reasonable_number_input_bounds("rainfall", 0.0, rainfall_default, monthly, forecast)
        humidity_min, humidity_max = reasonable_number_input_bounds("relative_humidity", 0.0, humidity_default, monthly, forecast)
        temp_min, temp_max = reasonable_number_input_bounds("temp_mid", 0.0, temp_default, monthly, forecast)
        cases_min, cases_max = reasonable_number_input_bounds("CHSO_cases", 0.0, 1.0, monthly)
        forecast_cases_proxy_values = combined_numeric_values("predicted_city_cases_proxy", forecast)
        if not forecast_cases_proxy_values.empty:
            cases_max = max(cases_max, float(forecast_cases_proxy_values.max()) * 1.25)

        st.markdown(f"### Target Month: {month_name_from_number(selected_month_num)} {selected_year_num}")
        climate_col1, climate_col2, climate_col3 = st.columns(3)
        with climate_col1:
            rainfall_now = st.number_input(
                "Target Month Rainfall",
                min_value=float(rainfall_min),
                max_value=float(rainfall_max),
                value=float(rainfall_default),
                step=1.0,
                format="%.4f",
            )
        with climate_col2:
            humidity_now = st.number_input(
                "Target Month Relative Humidity",
                min_value=float(humidity_min),
                max_value=float(humidity_max),
                value=float(humidity_default),
                step=0.1,
                format="%.4f",
            )
        with climate_col3:
            temp_now = st.number_input(
                "Target Month temp_mid",
                min_value=float(temp_min),
                max_value=float(temp_max),
                value=float(temp_default),
                step=0.1,
                format="%.4f",
            )

        st.markdown("### Recent Dengue Case History")
        if target_forecast_row is not None:
            default_cases_lag_1 = float(pd.to_numeric(pd.Series([target_forecast_row.get("cases_lag_1", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
            default_cases_lag_2 = float(pd.to_numeric(pd.Series([target_forecast_row.get("cases_lag_2", default_cases_lag_1)]), errors="coerce").fillna(default_cases_lag_1).iloc[0])
            default_cases_lag_3 = float(pd.to_numeric(pd.Series([target_forecast_row.get("cases_lag_3", default_cases_lag_2)]), errors="coerce").fillna(default_cases_lag_2).iloc[0])
        else:
            cases_series = numeric_series(monthly, "CHSO_cases").dropna()
            default_cases_lag_1 = float(cases_series.iloc[-1]) if len(cases_series) >= 1 else 0.0
            default_cases_lag_2 = float(cases_series.iloc[-2]) if len(cases_series) >= 2 else default_cases_lag_1
            default_cases_lag_3 = float(cases_series.iloc[-3]) if len(cases_series) >= 3 else default_cases_lag_2

        cases_max = max(cases_max, default_cases_lag_1, default_cases_lag_2, default_cases_lag_3, 1.0)
        case_col1, case_col2, case_col3 = st.columns(3)
        with case_col1:
            cases_lag_1 = st.number_input("Cases Last Month", min_value=0.0, max_value=float(cases_max), value=float(default_cases_lag_1), step=1.0)
        with case_col2:
            cases_lag_2 = st.number_input("Cases 2 Months Ago", min_value=0.0, max_value=float(cases_max), value=float(default_cases_lag_2), step=1.0)
        with case_col3:
            cases_lag_3 = st.number_input("Cases 3 Months Ago", min_value=0.0, max_value=float(cases_max), value=float(default_cases_lag_3), step=1.0)

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
        input_df = pd.DataFrame([input_values], columns=feature_columns)

        with st.expander("Show automatically prepared model inputs", expanded=False):
            st.dataframe(round_display_columns(input_df, feature_columns, 6), use_container_width=True)

        if st.button("Predict", type="primary"):
            try:
                pred, prob = safe_predict(model, input_df)
                result_col1, result_col2 = st.columns(2)
                result_col1.success(f"Predicted Class: {outbreak_label_from_binary(pred)}")
                result_col2.info(f"Predicted Outbreak Probability: {make_probability_label(prob)}")
                st.caption("0 = Non-outbreak month; 1 = Outbreak month. Probability is the model's estimated likelihood of the outbreak class.")

                st.subheader("Barangays with the Highest Predicted Risk")
                city_cases_proxy = float(input_values.get("cases_roll3_mean", 0.0)) * (1 + (0.0 if pd.isna(prob) else float(prob)))
                live_barangay_top3 = build_live_barangay_priority(barangay_risk_profile, selected_month_num, prob, city_cases_proxy)

                if live_barangay_top3 is not None and not live_barangay_top3.empty:
                    keep_cols = [
                        c
                        for c in [
                            "Barangay",
                            "overall_share",
                            "recent_share",
                            "seasonal_share",
                            "risk_score_raw",
                            "risk_score",
                            "predicted_outbreak_probability",
                            "predicted_city_cases_proxy",
                            "predicted_barangay_cases_proxy",
                            "predicted_barangay_label",
                        ]
                        if c in live_barangay_top3.columns
                    ]
                    st.dataframe(
                        round_display_columns(
                            live_barangay_top3[keep_cols],
                            [c for c in keep_cols if c != "Barangay" and c != "predicted_barangay_label"],
                            4,
                        ),
                        use_container_width=True,
                    )

                    fig_live_barangay = px.bar(
                        round_display_columns(live_barangay_top3, ["predicted_barangay_cases_proxy"], 2),
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
                    st.warning("barangay_risk_profile.csv is unavailable, so live barangay ranking cannot be generated.")
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.write("Check whether meta.json feature_columns match the trained best_model.joblib feature order.")

st.markdown("---")
st.caption("Baguio City Dengue Outbreak Forecast Dashboard")
