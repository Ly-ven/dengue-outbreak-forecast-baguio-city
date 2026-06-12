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

# Feature columns (match Colab export)
DEFAULT_FEATURE_COLS = [
    "rainfall", "relative_humidity", "temp_mid",
    "cases_lag_1", "cases_lag_2", "cases_lag_3",
    "rainfall_lag_1", "rainfall_lag_2", "rainfall_lag_3",
    "relative_humidity_lag_1", "relative_humidity_lag_2", "relative_humidity_lag_3",
    "temp_mid_lag_1", "temp_mid_lag_2", "temp_mid_lag_3",
    "cases_roll3_mean", "cases_roll3_max",
    "month_sin", "month_cos",
]

# Map artifact names to actual filenames from export
ARTIFACT_FILES = {
    "monthly":                  "monthly_data.csv",
    "model_comparison":         "model_comparison.csv",
    "auc_df":                   "model_auc.csv",
    "feature_importance":       "feature_importance.csv",
    "feature_sensitivity":      "feature_sensitivity.csv",
    "forecast":                 "forecast_5yr.csv",
    "barangay_monthly":         "barangay_monthly.csv",
    "top_barangay_monthly":     "top_barangay_monthly.csv",
    "top3_barangays_yearly":    "top3_barangays_yearly.csv",
    "top3_barangays_overall":   "top3_barangays_overall.csv",
    "test_predictions":         "test_predictions.csv",
    "climate_case_correlation": "climate_case_correlation.csv",
    "month_profile":            "month_profile.csv",
    "forecast_top3_barangays":  "forecast_top3_barangays.csv",
    "barangay_risk_profile":    "barangay_risk_profile.csv",
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
st.caption("Interactive dashboard for predicting dengue outbreaks in Baguio City using machine learning.")


# =========================
# LOADING HELPERS
# =========================
def load_csv(filename):
    path = ARTIFACTS_DIR / filename
    if path.exists():
        return pd.read_csv(path)
    return None


def load_json(filename):
    path = ARTIFACTS_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def load_model():
    path = ARTIFACTS_DIR / "best_model.joblib"
    if path.exists():
        return joblib.load(path)
    return None


@st.cache_data(show_spinner=False)
def load_all_artifacts():
    artifacts = {}
    for key, filename in ARTIFACT_FILES.items():
        artifacts[key] = load_csv(filename)
    artifacts["meta"] = load_json("meta.json")
    artifacts["feature_columns"] = load_json("feature_columns.json")
    return artifacts


artifacts = load_all_artifacts()
model = load_model()
meta = artifacts.get("meta")
feature_cols_from_meta = artifacts.get("feature_columns", DEFAULT_FEATURE_COLS)

# Extract dataframes
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


# =========================
# SIDEBAR
# =========================
st.sidebar.header("Dashboard Status")

if monthly is not None:
    st.sidebar.success("Artifacts loaded successfully")
    if meta:
        st.sidebar.info(f"Best Model: {meta.get('best_model', 'Unknown')}")
        threshold_val = meta.get("outbreak_threshold_cases", "N/A")
        if isinstance(threshold_val, (int, float)):
            st.sidebar.info(f"Outbreak Threshold: {threshold_val:.2f} cases")
    if forecast is not None:
        st.sidebar.info(f"Forecast Period: {meta.get('forecast_period', '2027-2031')}")
else:
    st.sidebar.error("Artifacts not found. Place the 'artifacts' folder in the same directory as this app.")
    st.sidebar.markdown("Required files:")
    for filename in ARTIFACT_FILES.values():
        st.sidebar.code(f"artifacts/{filename}")
    st.sidebar.code("artifacts/best_model.joblib")
    st.sidebar.code("artifacts/meta.json")
    st.sidebar.code("artifacts/feature_columns.json")
    st.stop()


# =========================
# HELPERS
# =========================
def month_name_from_number(month_num):
    names = {1:"Jan",2:"Feb",3:"Mar",4:"Apr",5:"May",6:"Jun",
             7:"Jul",8:"Aug",9:"Sep",10:"Oct",11:"Nov",12:"Dec"}
    return names.get(int(month_num), str(month_num))


def safe_metric_value(value, decimals=2):
    try:
        if pd.isna(value):
            return "N/A"
        return f"{float(value):.{decimals}f}"
    except Exception:
        return "N/A"


def round_df(df, columns, decimals=2):
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(decimals)
    return df


def parse_dates(df, key):
    if df is not None and "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df


# Parse dates
for key in DATE_COLUMNS:
    df = locals().get(key)
    if df is not None:
        locals()[key] = parse_dates(df, key)


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
        # Metrics row
        total_months = len(monthly)
        total_cases = int(pd.to_numeric(monthly["CHSO_cases"], errors="coerce").fillna(0).sum()) if "CHSO_cases" in monthly.columns else 0
        avg_cases = pd.to_numeric(monthly["CHSO_cases"], errors="coerce").mean() if "CHSO_cases" in monthly.columns else np.nan
        outbreak_months = int(pd.to_numeric(monthly["outbreak"], errors="coerce").fillna(0).sum()) if "outbreak" in monthly.columns else 0

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Months", total_months)
        col2.metric("Total CHSO Cases", f"{total_cases:,}")
        col3.metric("Average Monthly Cases", safe_metric_value(avg_cases))
        col4.metric("Outbreak Months", outbreak_months)

        # Model definition
        if meta:
            st.info(f"**Outbreak Definition:** {meta.get('outbreak_definition', 'Monthly CHSO cases >= 75th percentile')}")

        # Time series chart - full width
        st.subheader("Monthly Dengue Cases Trend")
        if "Date" in monthly.columns and "CHSO_cases" in monthly.columns:
            if "DOH_cases" in monthly.columns:
                trend_long = monthly[["Date", "CHSO_cases", "DOH_cases"]].melt(id_vars="Date", var_name="Source", value_name="Cases")
                fig_line = px.line(trend_long, x="Date", y="Cases", color="Source", markers=True,
                                   title="CHSO vs DOH Monthly Dengue Counts")
            else:
                fig_line = px.line(monthly, x="Date", y="CHSO_cases", markers=True,
                                   title="Monthly Dengue Cases in Baguio City (CHSO)")
            st.plotly_chart(fig_line, use_container_width=True)

        # Two column layout for heatmap and correlations
        st.subheader("Seasonal Patterns")
        col_left, col_right = st.columns(2)

        with col_left:
            # Heatmap
            if {"Year", "Month", "CHSO_cases"}.issubset(monthly.columns):
                heat = monthly.pivot_table(index="Year", columns="Month", values="CHSO_cases", aggfunc="sum")
                fig_heat = px.imshow(heat, text_auto=True, aspect="auto", color_continuous_scale="YlOrRd",
                                     title="Year-Month Heatmap of Dengue Cases")
                fig_heat.update_xaxes(title="Month", tickvals=list(range(1, 13)), 
                                      ticktext=[month_name_from_number(m) for m in range(1, 13)])
                fig_heat.update_yaxes(title="Year")
                st.plotly_chart(fig_heat, use_container_width=True)

        with col_right:
            # Climate correlation bar chart
            if climate_case_correlation is not None and not climate_case_correlation.empty:
                fig_corr = px.bar(
                    climate_case_correlation,
                    x="feature", 
                    y="pearson_corr_with_CHSO_cases",
                    text="pearson_corr_with_CHSO_cases",
                    title="Climate-Case Correlation",
                    labels={"feature": "Climate Variable", "pearson_corr_with_CHSO_cases": "Pearson Correlation"}
                )
                fig_corr.update_traces(texttemplate="%{text:.4f}", textposition="outside")
                fig_corr.update_yaxes(range=[0, 0.3])
                st.plotly_chart(fig_corr, use_container_width=True)

        # Monthly profile chart
        if month_profile is not None and not month_profile.empty:
            st.subheader("Average Monthly Cases by Month")
            fig_month = px.bar(
                month_profile,
                x="MonthName",
                y="CHSO_cases",
                text="CHSO_cases",
                title="Average CHSO Cases by Month",
                labels={"MonthName": "Month", "CHSO_cases": "Average Cases"}
            )
            fig_month.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(fig_month, use_container_width=True)

            # Optional: Expandable table for month profile data
            with st.expander("View Monthly Profile Data"):
                display_cols = ["MonthName", "CHSO_cases", "rainfall", "relative_humidity", "temp_mid"]
                available_cols = [c for c in display_cols if c in month_profile.columns]
                st.dataframe(round_df(month_profile[available_cols], ["CHSO_cases", "rainfall", "relative_humidity", "temp_mid"], 2), 
                            use_container_width=True)

        # Climate profile comparison
        if {"outbreak", "rainfall", "relative_humidity", "temp_mid"}.issubset(monthly.columns):
            st.subheader("Climate Profile: Outbreak vs Non-outbreak Months")
            climate_profile = monthly.groupby("outbreak", as_index=False)[["rainfall", "relative_humidity", "temp_mid"]].mean()
            climate_profile["Outbreak Status"] = climate_profile["outbreak"].map({0: "Non-outbreak", 1: "Outbreak"})
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
                labels={"Average Value": "Average Value", "Climate Variable": "Climate Variable"}
            )
            fig_climate.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            st.plotly_chart(fig_climate, use_container_width=True)


# =========================
# TAB 2: BARANGAY ANALYTICS
# =========================
with tab2:
    st.header("Barangay Analytics")

    if top_barangay_monthly is not None and not top_barangay_monthly.empty:
        st.subheader("Barangay with Highest Monthly Dengue Cases")
        st.dataframe(top_barangay_monthly, use_container_width=True)

    if top3_barangays_yearly is not None and not top3_barangays_yearly.empty:
        st.subheader("Three Highest per Year")
        st.dataframe(top3_barangays_yearly, use_container_width=True)

    if top3_barangays_overall is not None and not top3_barangays_overall.empty:
        st.subheader("Three Highest Overall")
        st.dataframe(top3_barangays_overall, use_container_width=True)

    if barangay_monthly is not None and not barangay_monthly.empty:
        st.subheader("Barangay Monthly Records")
        st.dataframe(barangay_monthly, use_container_width=True)


# =========================
# TAB 3: MODEL RESULTS
# =========================
with tab3:
    st.header("Model Results")

    if meta:
        st.success(f"Selected Model: {meta.get('best_model', 'Unknown')}")

    if model_comparison is not None and not model_comparison.empty:
        st.subheader("Model Performance Comparison")
        display_cols = [c for c in ["Model", "Accuracy", "F1 Score", "Precision", "Recall", "AUC (Supplementary)", "Reliability (Brier)"] if c in model_comparison.columns]
        st.dataframe(round_df(model_comparison[display_cols], ["Accuracy", "F1 Score", "Precision", "Recall", "AUC (Supplementary)", "Reliability (Brier)"], 4), use_container_width=True)
        
        # Show best model accuracy
        if "Model" in model_comparison.columns and "Accuracy" in model_comparison.columns:
            best_row = model_comparison.loc[model_comparison["Accuracy"].idxmax()]
            st.metric("Best Model Accuracy", f"{best_row['Accuracy']:.4f}", delta=f"Model: {best_row['Model']}")

    if test_predictions is not None and not test_predictions.empty:
        st.subheader("Month-by-Month Test Predictions")
        correct = int(pd.to_numeric(test_predictions["is_correct"], errors="coerce").fillna(0).sum()) if "is_correct" in test_predictions.columns else 0
        accuracy = correct / len(test_predictions) if len(test_predictions) > 0 else 0
        st.metric("Test Set Accuracy", f"{accuracy:.4f}", delta=f"{correct} / {len(test_predictions)} correct")
        st.dataframe(test_predictions, use_container_width=True)


# =========================
# TAB 4: FEATURE TRANSPARENCY
# =========================
with tab4:
    st.header("Feature Transparency")

    if feature_importance is not None and not feature_importance.empty:
        st.subheader("Feature Importance (Permutation Importance)")
        st.dataframe(round_df(feature_importance, ["importance_mean", "importance_std"], 6), use_container_width=True)
        
        if "feature" in feature_importance.columns and "importance_mean" in feature_importance.columns:
            fig_imp = px.bar(feature_importance.head(10), x="importance_mean", y="feature", orientation="h",
                             title="Top 10 Most Important Features")
            st.plotly_chart(fig_imp, use_container_width=True)

    if feature_sensitivity is not None and not feature_sensitivity.empty:
        st.subheader("Sensitivity Analysis (+10% Climate Change)")
        st.dataframe(round_df(feature_sensitivity, ["change_in_probability", "percent_change"], 6), use_container_width=True)


# =========================
# TAB 5: FORECAST & LIVE PREDICTION
# =========================
with tab5:
    st.header("Forecast & Live Prediction")

    if forecast is not None and not forecast.empty:
        st.subheader("Five-Year Forecast (2027-2031)")
        st.dataframe(round_df(forecast.head(30), ["predicted_outbreak_probability", "predicted_city_cases_proxy"], 4), use_container_width=True)

        if "Date" in forecast.columns and "predicted_outbreak_probability" in forecast.columns:
            fig_forecast = px.line(forecast, x="Date", y="predicted_outbreak_probability", markers=True,
                                   title="5-Year Forecasted Outbreak Probability")
            st.plotly_chart(fig_forecast, use_container_width=True)

        if {"Year", "Month", "predicted_outbreak_probability"}.issubset(forecast.columns):
            forecast_heat = forecast.pivot_table(index="Year", columns="Month", values="predicted_outbreak_probability")
            forecast_heat.columns = [month_name_from_number(m) for m in forecast_heat.columns]
            fig_heat = px.imshow(forecast_heat, text_auto=".3f", aspect="auto", color_continuous_scale="Reds",
                                 title="Forecast Heatmap - Outbreak Probability")
            fig_heat.update_xaxes(title="Month")
            fig_heat.update_yaxes(title="Year")
            st.plotly_chart(fig_heat, use_container_width=True)

    if forecast_top3_barangays is not None and not forecast_top3_barangays.empty:
        st.subheader("Top 3 Barangays - Forecast Months")
        st.dataframe(forecast_top3_barangays, use_container_width=True)

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
            selected_month = st.selectbox("Month", list(range(1, 13)), format_func=lambda x: f"{x} - {month_name_from_number(x)}", index=5)

        if month_profile is not None:
            month_data = month_profile[month_profile["Month"] == selected_month]
            default_rain = float(month_data["rainfall"].iloc[0]) if not month_data.empty else 100.0
            default_humidity = float(month_data["relative_humidity"].iloc[0]) if not month_data.empty else 80.0
            default_temp = float(month_data["temp_mid"].iloc[0]) if not month_data.empty else 25.0
        else:
            default_rain, default_humidity, default_temp = 100.0, 80.0, 25.0

        forecast_row = None
        if forecast is not None:
            mask = (forecast["Year"] == selected_year) & (forecast["Month"] == selected_month)
            if mask.any():
                forecast_row = forecast[mask].iloc[0]

        if forecast_row is not None:
            default_rain = float(forecast_row.get("rainfall", default_rain))
            default_humidity = float(forecast_row.get("relative_humidity", default_humidity))
            default_temp = float(forecast_row.get("temp_mid", default_temp))

        st.markdown(f"**Target Month: {month_name_from_number(selected_month)} {selected_year}**")
        c1, c2, c3 = st.columns(3)
        with c1:
            rainfall = st.slider("Rainfall (mm)", 0.0, 1500.0, default_rain, step=10.0)
        with c2:
            humidity = st.slider("Relative Humidity (%)", 60.0, 100.0, default_humidity, step=1.0)
        with c3:
            temp = st.slider("Temperature (C)", 10.0, 35.0, default_temp, step=0.5)

        st.markdown("**Recent Dengue Cases**")
        if monthly is not None and "CHSO_cases" in monthly.columns:
            cases_series = pd.to_numeric(monthly["CHSO_cases"], errors="coerce").dropna()
            default_lag1 = float(cases_series.iloc[-1]) if len(cases_series) >= 1 else 50.0
            default_lag2 = float(cases_series.iloc[-2]) if len(cases_series) >= 2 else default_lag1
            default_lag3 = float(cases_series.iloc[-3]) if len(cases_series) >= 3 else default_lag2
        else:
            default_lag1 = default_lag2 = default_lag3 = 50.0

        if forecast_row is not None:
            default_lag1 = float(forecast_row.get("cases_lag_1", default_lag1))
            default_lag2 = float(forecast_row.get("cases_lag_2", default_lag2))
            default_lag3 = float(forecast_row.get("cases_lag_3", default_lag3))

        d1, d2, d3 = st.columns(3)
        with d1:
            cases_lag1 = st.number_input("Cases - Last Month", min_value=0, value=int(default_lag1), step=10)
        with d2:
            cases_lag2 = st.number_input("Cases - 2 Months Ago", min_value=0, value=int(default_lag2), step=10)
        with d3:
            cases_lag3 = st.number_input("Cases - 3 Months Ago", min_value=0, value=int(default_lag3), step=10)

        lag_periods = [(selected_year, selected_month - i) for i in range(1, 4)]
        for i, (y, m) in enumerate(lag_periods):
            if m <= 0:
                lag_periods[i] = (y - 1, m + 12)

        def get_lag_value(periods, idx, col, default):
            y, m = periods[idx]
            if forecast is not None:
                mask = (forecast["Year"] == y) & (forecast["Month"] == m)
                if mask.any():
                    return float(forecast[mask].iloc[0].get(col, default))
            if month_profile is not None:
                month_data = month_profile[month_profile["Month"] == m]
                if not month_data.empty:
                    return float(month_data[col].iloc[0])
            return default

        features = {
            "rainfall": rainfall,
            "relative_humidity": humidity,
            "temp_mid": temp,
            "cases_lag_1": cases_lag1,
            "cases_lag_2": cases_lag2,
            "cases_lag_3": cases_lag3,
            "rainfall_lag_1": get_lag_value(lag_periods, 0, "rainfall", 100.0),
            "rainfall_lag_2": get_lag_value(lag_periods, 1, "rainfall", 100.0),
            "rainfall_lag_3": get_lag_value(lag_periods, 2, "rainfall", 100.0),
            "relative_humidity_lag_1": get_lag_value(lag_periods, 0, "relative_humidity", 80.0),
            "relative_humidity_lag_2": get_lag_value(lag_periods, 1, "relative_humidity", 80.0),
            "relative_humidity_lag_3": get_lag_value(lag_periods, 2, "relative_humidity", 80.0),
            "temp_mid_lag_1": get_lag_value(lag_periods, 0, "temp_mid", 25.0),
            "temp_mid_lag_2": get_lag_value(lag_periods, 1, "temp_mid", 25.0),
            "temp_mid_lag_3": get_lag_value(lag_periods, 2, "temp_mid", 25.0),
            "cases_roll3_mean": np.mean([cases_lag1, cases_lag2, cases_lag3]),
            "cases_roll3_max": np.max([cases_lag1, cases_lag2, cases_lag3]),
            "month_sin": np.sin(2 * np.pi * selected_month / 12.0),
            "month_cos": np.cos(2 * np.pi * selected_month / 12.0),
        }

        if st.button("Predict Outbreak", type="primary"):
            input_df = pd.DataFrame([{k: features.get(k, 0) for k in feature_cols_from_meta}])
            pred = int(model.predict(input_df)[0])
            prob = float(model.predict_proba(input_df)[0][1]) if hasattr(model, "predict_proba") else np.nan

            col_a, col_b = st.columns(2)
            col_a.success(f"Prediction: {'OUTBREAK' if pred == 1 else 'NON-OUTBREAK'}")
            col_b.info(f"Outbreak Probability: {prob:.4f}" if not pd.isna(prob) else "Probability not available")

            st.caption("Threshold for outbreak classification is the 75th percentile of historical monthly cases.")

st.markdown("---")
st.caption("Baguio City Dengue Outbreak Forecast Dashboard | Powered by Machine Learning")
