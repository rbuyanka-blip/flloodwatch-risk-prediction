from pathlib import Path
import importlib.util
import json
import warnings

import numpy as np
import pandas as pd
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "flood_risk_model.joblib"
HELPER_PATH = APP_DIR / "predict_flood_risk.py"
METRICS_PATH = APP_DIR / "model_metrics.json"

st.set_page_config(page_title="FloodWatch | Telangana", page_icon="🌧️", layout="wide")

@st.cache_resource
def load_helper():
    if not HELPER_PATH.exists():
        raise FileNotFoundError(f"Missing helper file: {HELPER_PATH.name}")
    spec = importlib.util.spec_from_file_location("project_predict_flood_risk", HELPER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

st.title("🌧️ FloodWatch")
st.subheader("Rainfall + water-level based flood-risk prediction")
st.write(
    "This app uses the project's saved Random Forest classifier, probability calibrator, "
    "thresholds, and `predict_from_history` feature-building helper."
)
st.warning(
    "Research prototype only. It is not an official warning service and must not be used "
    "for emergency decisions. The evaluation reports many false alarms."
)

if not MODEL_PATH.exists() or not HELPER_PATH.exists():
    st.error("Required model/helper file is missing. Keep app.py, flood_risk_model.joblib, and predict_flood_risk.py together.")
    st.stop()

with st.sidebar:
    st.header("Model information")
    st.success("Model bundle found")
    st.success("Prediction helper found")
    try:
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            st.write("Selected model:", metrics.get("selected_model", "Random Forest"))
            st.write("Test ROC-AUC:", f"{metrics.get('test_roc_auc', float('nan')):.3f}")
            st.write("Test PR-AUC:", f"{metrics.get('test_pr_auc', float('nan')):.3f}")
    except Exception:
        st.caption("Metrics file could not be read.")

st.markdown("### 1. Provide the latest 7+ days of hourly history")
st.write(
    "Upload a CSV with one row per hour for a single gauge/catchment. It needs a timestamp column, "
    "hourly catchment-average rainfall in millimetres, and water level in metres. "
    "The app maps your selected columns to the exact names required by the project helper."
)
st.caption("Required history fields: timestamp, rainfall (mm), water level (m). At least 168 hourly rows are required.")

uploaded = st.file_uploader("Upload hourly history CSV", type=["csv"])

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        if raw.empty:
            st.error("The uploaded CSV is empty.")
            st.stop()
        st.write("Preview of uploaded data")
        st.dataframe(raw.head(8), use_container_width=True)

        columns = list(raw.columns)
        default_ts = next((i for i, c in enumerate(columns) if str(c).lower() in
                           ["timestamp", "datetime", "date_time", "time", "date", "t"]), 0)
        default_rain = next((i for i, c in enumerate(columns) if str(c).lower() in
                             ["rain_mm", "rainfall_mm", "rainfall", "rain_mm_hour", "mm"]), 1 if len(columns) > 1 else 0)
        default_level = next((i for i, c in enumerate(columns) if str(c).lower() in
                              ["level_m", "water_level_m", "water_level", "level", "lvl"]), min(2, len(columns)-1))

        st.markdown("#### Match the CSV columns")
        c1, c2, c3 = st.columns(3)
        with c1:
            ts_col = st.selectbox("Timestamp column", columns, index=default_ts)
        with c2:
            rain_col = st.selectbox("Rainfall column (mm)", columns, index=default_rain)
        with c3:
            level_col = st.selectbox("Water-level column (m)", columns, index=default_level)

        if len({ts_col, rain_col, level_col}) < 3:
            st.error("Choose three different columns for timestamp, rainfall, and water level.")
            st.stop()

        history = raw[[ts_col, rain_col, level_col]].copy()
        history[ts_col] = pd.to_datetime(history[ts_col], errors="coerce")
        history[rain_col] = pd.to_numeric(history[rain_col], errors="coerce")
        history[level_col] = pd.to_numeric(history[level_col], errors="coerce")
        history = history.dropna(subset=[ts_col]).sort_values(ts_col)
        history = history.rename(columns={rain_col: "Rain_mm", level_col: "Level_m"})
        history = history[[ts_col, "Rain_mm", "Level_m"]].set_index(ts_col)
        history.index.name = "Timestamp"
        history = history[~history.index.duplicated(keep="last")]

        st.write(f"Valid timestamped rows: **{len(history):,}**")
        if len(history) < 168:
            st.error(f"Only {len(history)} rows found. The project helper requires at least 168 hourly rows (7 days).")
            st.stop()

        st.markdown("### 2. Set location metadata")
        c1, c2, c3 = st.columns(3)
        with c1:
            lat = st.number_input("Gauge latitude", value=19.359175, format="%.6f")
        with c2:
            lon = st.number_input("Gauge longitude", value=79.306053, format="%.6f")
        with c3:
            n_stations = st.number_input("Rainfall stations represented", min_value=1, max_value=38, value=1)

        st.caption("Use the actual gauge coordinates and the number of rainfall stations represented by the catchment-average rainfall.")

        if st.button("Predict flood risk", type="primary"):
            try:
                # Keep the uploaded model bundle and the project's own feature builder together.
                helper = load_helper()
                with warnings.catch_warnings():
                    warnings.simplefilter("default")
                    result = helper.predict_from_history(
                        history,
                        lat=float(lat),
                        lon=float(lon),
                        n_rain_stations=int(n_stations),
                        model_path=str(MODEL_PATH),
                    )

                prob = float(result["probability"])
                risk = str(result["risk_level"])
                meaning = str(result.get("meaning", "Water level rises >= 0.5 m within 6 hours"))

                st.markdown("### 3. Prediction result")
                a, b = st.columns(2)
                with a:
                    st.metric("Predicted risk category", risk)
                with b:
                    st.metric("Calibrated model probability", f"{prob:.2%}")

                if risk == "High":
                    st.error("High model-indicated risk — this is not an official alert.")
                elif risk == "Moderate":
                    st.warning("Moderate model-indicated risk — verify against trusted local monitoring.")
                else:
                    st.info("Low model-indicated risk — this does not guarantee safety.")

                st.write("**Target definition:**", meaning)
                st.write("**History ending at:**", str(history.index.max()))
                with st.expander("Technical details"):
                    st.json(result)
            except Exception as exc:
                st.error("Prediction could not be completed.")
                st.exception(exc)
                st.info(
                    "Check that the CSV contains hourly observations, valid units, and enough data. "
                    "The uploaded history must represent one gauge/catchment, not the model's precomputed training feature table."
                )
    except Exception as exc:
        st.error(f"Could not prepare the CSV: {exc}")

st.divider()
st.markdown("### Evaluation caveat")
st.write(
    "The supplied evaluation reports ROC-AUC ≈ 0.891 and PR-AUC ≈ 0.122, with a test event rate of about 0.57%. "
    "At the saved moderate threshold, reported precision is about 3.6%; at the high threshold, about 8.1%. "
    "These results indicate substantial false-alarm risk, so this app is for demonstration/research only."
)
