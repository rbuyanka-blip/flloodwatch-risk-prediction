# FloodWatch — integrated with your uploaded project files

This package includes the Streamlit app and your uploaded trained model/helper files.

## Files
- `app.py` — Streamlit web app
- `flood_risk_model.joblib` — your uploaded trained model bundle (Random Forest classifier + calibrator + thresholds)
- `predict_flood_risk.py` — your uploaded project prediction helper
- `model_metrics.json` — your uploaded evaluation metrics
- `requirements.txt` — dependencies

## Run on a computer
Use Python 3.13 if possible, to match the Colab notebook's saved-model environment more closely.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit, usually `http://localhost:8501`.

## Input CSV format
Upload at least 168 hourly rows for one gauge/catchment. Choose the timestamp, rainfall, and water-level columns in the app. The helper requires:
- hourly timestamp as a datetime index
- `Rain_mm`: catchment-average rainfall in mm for each hour
- `Level_m`: water level in metres

Rainfall must represent the catchment average for the gauge, not unrelated stations or an arbitrary sum. Use actual latitude/longitude and the number of rainfall stations represented.

The precomputed training dataset `ml_dataset_flood_risk.csv.gz` is intentionally not included in the app package because it contains engineered training features, not the raw hourly history format expected by `predict_from_history`.

## Model note
The saved bundle is the existing trained model; this package does not retrain it. The model file was serialized with scikit-learn 1.8.0 according to the notebook warning, so the requirements pin that version. Loading serialized sklearn models across versions can be unsupported; use the same version as training.

## Deploy online
After testing locally, deploy this folder through Streamlit Community Cloud or another Python hosting service. Keep model files in a private repository if they should not be public. Do not treat predictions as official warnings; metrics show substantial false-alarm risk.
