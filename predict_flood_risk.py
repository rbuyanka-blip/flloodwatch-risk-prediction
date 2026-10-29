"""Flood-risk prediction helper for the Streamlit app.

Usage
-----
from predict_flood_risk import predict_from_history, predict_from_features

history: hourly DataFrame (>= 168 rows = 7 days, ending at 'now', DatetimeIndex) with
   Rain_mm : catchment-average rainfall in mm for that hour (0 when dry)
   Level_m : water level (any consistent datum), NaN allowed for gaps
Returns dict: probability (calibrated), risk_level ('Low'/'Moderate'/'High'), target description.
"""
import numpy as np, pandas as pd, joblib

_cache = {}
def _bundle(path):
    if path not in _cache: _cache[path] = joblib.load(path)
    return _cache[path]

def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p)).reshape(-1, 1)

def build_features(history, lat, lon, n_rain_stations=1):
    h = history.sort_index().copy()
    if len(h) < 168: raise ValueError('Need at least 168 hourly rows (7 days) of history.')
    h = h.asfreq('h')
    rn = h['Rain_mm']; xi = h['Level_m'].interpolate(limit=3, limit_area='inside')
    xs = xi.rolling(3, min_periods=2).median()
    ix = h.index; f = {}
    f['Rain_1H'] = rn.iloc[-1]
    for k in (3, 6, 12, 24, 72): f[f'Rain_{k}H'] = rn.iloc[-k:].sum()
    f['Rain_7D'] = rn.iloc[-168:].sum()
    f['Rain_Max1H_24H'] = rn.iloc[-24:].max()
    wet = np.where(rn.fillna(0).values > 0)[0]
    f['Hours_Since_Rain'] = min(len(rn) - 1 - wet[-1], 72) if len(wet) else 72
    f['N_Rain_Stations'] = n_rain_stations
    for k in (1, 3, 6, 12, 24): f[f'Level_Change_{k}H'] = xs.iloc[-1] - xs.iloc[-1 - k]
    f['Level_Std_24H'] = xs.iloc[-24:].std()
    f['Level_vs_7D_Median'] = xs.iloc[-1] - xs.iloc[-168:].median()
    t = ix[-1]
    f['Month_Sin'], f['Month_Cos'] = np.sin(2*np.pi*t.month/12), np.cos(2*np.pi*t.month/12)
    f['Hour_Sin'], f['Hour_Cos'] = np.sin(2*np.pi*t.hour/24), np.cos(2*np.pi*t.hour/24)
    f['Latitude'], f['Longitude'] = lat, lon
    return pd.DataFrame([f])

def predict_from_features(features, model_path='flood_risk_model.joblib'):
    b = _bundle(model_path)
    X = features[b['features']]
    if X.isna().any().any(): raise ValueError('Missing values in inputs: %s' % list(X.columns[X.isna().any()]))
    raw = b['model'].predict_proba(X)[:, 1]
    p = b['calibrator'].predict_proba(_logit(raw))[:, 1]
    th = b['thresholds']
    lvl = np.where(p >= th['high'], 'High', np.where(p >= th['moderate'], 'Moderate', 'Low'))
    return {'probability': float(p[0]), 'risk_level': str(lvl[0]), 'meaning': b['target']}

def predict_from_history(history, lat, lon, n_rain_stations=1, model_path='flood_risk_model.joblib'):
    return predict_from_features(build_features(history, lat, lon, n_rain_stations), model_path)
