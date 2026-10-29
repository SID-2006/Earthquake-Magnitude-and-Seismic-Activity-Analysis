import json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import streamlit as st

DATA_PATH = Path("earthquake_data_cleaned.csv")
CLASSIFIER_PATH = Path("best_earthquake_classifier.joblib")
REGRESSOR_PATH = Path("earthquake_magnitude_regressor.joblib")
REGRESSOR_LR_PATH = Path("earthquake_magnitude_linear_regressor.joblib")
REGRESSOR_RF_PATH = Path("earthquake_magnitude_rf_regressor.joblib")
REGRESSOR_DT_PATH = Path("earthquake_magnitude_dt_regressor.joblib")
CLUSTERER_PATH = Path("earthquake_spatial_clusterer.joblib")
REGIONAL_MODELS_PATH = Path("regional_cluster_models.joblib")
REGIONAL_METRICS_PATH = Path("regional_cluster_metrics.json")
METRICS_PATH = Path("model_metrics.json")
STATS_PATH = Path("statistical_inference_results.json")

# Preset Hotspots for Quick Diagnostics
PRESET_HOTSPOTS = {
    "Tokyo / Kanto (Japan)": {
        "depth": 35.0, "latitude": 35.6895, "longitude": 139.6917, "month": 3, "hour": 14,
        "region": "Ring of Fire - Subduction Zone", "risk_context": "High risk of deep megathrust and crustal events"
    },
    "San Andreas Fault (California)": {
        "depth": 8.5, "latitude": 37.7749, "longitude": -122.4194, "month": 4, "hour": 5,
        "region": "North American / Pacific Plate Boundary", "risk_context": "Strike-slip fault, shallow rupture depth"
    },
    "Sumatra Megathrust (Indonesia)": {
        "depth": 30.0, "latitude": 3.1764, "longitude": 95.9814, "month": 12, "hour": 7,
        "region": "Sunda Trench", "risk_context": "Major tsunami-generating subduction interface"
    },
    "Valdivia / Bio-Bio (Chile)": {
        "depth": 25.0, "latitude": -39.8142, "longitude": -73.2459, "month": 5, "hour": 19,
        "region": "Nazca / South American Subduction", "risk_context": "Historically produced largest recorded earthquakes"
    },
    "Himalayan Thrust (Nepal)": {
        "depth": 15.0, "latitude": 27.7172, "longitude": 85.3240, "month": 4, "hour": 11,
        "region": "Indian / Eurasian Continental Collision", "risk_context": "Intense crustal compression and shallow focus"
    },
    "Reykjanes Volcanic Rift (Iceland)": {
        "depth": 5.0, "latitude": 63.8500, "longitude": -22.4300, "month": 11, "hour": 17,
        "region": "Mid-Atlantic Ridge Divergent Margin", "risk_context": "Shallow volcanic seismic swarms and magma intrusion"
    },
    "Tonga Deep Trench": {
        "depth": 550.0, "latitude": -21.1789, "longitude": -175.1982, "month": 9, "hour": 22,
        "region": "Tonga-Kermadec Subduction Zone", "risk_context": "Deep focus Wadati-Benioff seismicity"
    }
}

@st.cache_resource(show_spinner=False)
def load_models():
    """Loads pre-trained classification, regression, and regional cluster models."""
    if not CLASSIFIER_PATH.exists():
        raise FileNotFoundError(f"Model file '{CLASSIFIER_PATH}' not found. Run train_models.py first.")
    
    clf = joblib.load(CLASSIFIER_PATH)
    reg_rf = joblib.load(REGRESSOR_RF_PATH) if REGRESSOR_RF_PATH.exists() else (joblib.load(REGRESSOR_PATH) if REGRESSOR_PATH.exists() else None)
    reg_lr = joblib.load(REGRESSOR_LR_PATH) if REGRESSOR_LR_PATH.exists() else None
    reg_dt = joblib.load(REGRESSOR_DT_PATH) if REGRESSOR_DT_PATH.exists() else None
    clusterer = joblib.load(CLUSTERER_PATH) if CLUSTERER_PATH.exists() else None
    regional_models = joblib.load(REGIONAL_MODELS_PATH) if REGIONAL_MODELS_PATH.exists() else {}
    
    return {
        "classifier": clf,
        "regressor_rf": reg_rf,
        "regressor_lr": reg_lr,
        "regressor_dt": reg_dt,
        "regressor": reg_rf,
        "clusterer": clusterer,
        "regional_models": regional_models
    }

@st.cache_data(show_spinner=False)
def load_dataset():
    """Loads and caches the cleaned earthquake dataset."""
    if not DATA_PATH.exists():
        return None
    df = pd.read_csv(DATA_PATH)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
    if "high_magnitude" not in df.columns and "mag" in df.columns:
        df["high_magnitude"] = (df["mag"] >= 4.0).astype(int)
    return df

def load_metrics():
    """Loads precomputed model performance and dataset summary metrics."""
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_regional_metrics():
    """Loads precomputed regional cluster metrics and tectonic chunk profiles."""
    if REGIONAL_METRICS_PATH.exists():
        with open(REGIONAL_METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def load_statistical_results():
    """Loads precomputed hypothesis testing and inferential statistics."""
    if STATS_PATH.exists():
        with open(STATS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def predict_tectonic_chunk(lat: float, lon: float, clusterer=None, regional_metrics=None):
    """
    Maps geographic coordinates to the nearest Earth Tectonic Cluster / Spatial Chunk.
    Uses unit-sphere 3D Cartesian coordinates to prevent boundary distortion at +-180 longitude.
    """
    if clusterer is None and CLUSTERER_PATH.exists():
        clusterer = joblib.load(CLUSTERER_PATH)
    if regional_metrics is None and REGIONAL_METRICS_PATH.exists():
        with open(REGIONAL_METRICS_PATH, "r", encoding="utf-8") as f:
            regional_metrics = json.load(f)
            
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    cart_x = np.cos(lat_rad) * np.cos(lon_rad)
    cart_y = np.cos(lat_rad) * np.sin(lon_rad)
    cart_z = np.sin(lat_rad)

    coords_df = pd.DataFrame([[cart_x, cart_y, cart_z]], columns=["cart_x", "cart_y", "cart_z"])
    cid = int(clusterer.predict(coords_df)[0])
    
    c_info = regional_metrics.get(str(cid), {
        "cluster_id": cid,
        "name": f"Tectonic Zone {cid}",
        "tectonic_type": "Tectonic Plate Margin",
        "mean_mag": 4.0,
        "max_mag": 7.0,
        "high_mag_ratio": 0.5,
        "mean_depth": 35.0
    })
    
    return cid, c_info

def predict_area_wise_magnitude(lat: float, lon: float, depth: float, month: int = 1, hour: int = 12, models_dict=None, regional_metrics=None):
    """
    Performs specialized Area-Wise Magnitude Prediction by:
    1. Identifying the local Tectonic Cluster Chunk.
    2. Invoking the localized regional expert Random Forest.
    3. Invoking the global Random Forest baseline.
    4. Producing a consensus blended estimate with regional uncertainty bounds.
    """
    if models_dict is None:
        models_dict = load_models()
    if regional_metrics is None:
        regional_metrics = load_regional_metrics()
        
    clusterer = models_dict.get("clusterer")
    reg_rf = models_dict.get("regressor_rf")
    reg_models = models_dict.get("regional_models", {})
    clf = models_dict.get("classifier")

    cid, c_info = predict_tectonic_chunk(lat, lon, clusterer, regional_metrics)
    
    features_df = pd.DataFrame(
        [[depth, lat, lon, month, hour]],
        columns=["depth", "latitude", "longitude", "month", "hour"]
    )
    
    # 1. Global prediction
    global_pred = float(reg_rf.predict(features_df)[0]) if reg_rf else float(c_info.get("mean_mag", 3.88))
    
    # 2. Local regional expert prediction
    local_model = reg_models.get(str(cid))
    if local_model is not None:
        local_pred = float(local_model.predict(features_df)[0])
    else:
        local_pred = global_pred
        
    # 3. Consensus blend: 70% regional expert + 30% global model
    hybrid_pred = float(0.70 * local_pred + 0.30 * global_pred)
    
    # 4. Classification probability
    prob_high = float(clf.predict_proba(features_df)[0, 1]) if clf else float(c_info.get("high_mag_ratio", 0.5))
    
    return {
        "cluster_id": cid,
        "cluster_name": c_info.get("name", f"Cluster {cid}"),
        "tectonic_type": c_info.get("tectonic_type", "Plate Boundary"),
        "cluster_info": c_info,
        "local_pred": round(local_pred, 2),
        "global_pred": round(global_pred, 2),
        "hybrid_pred": round(hybrid_pred, 2),
        "prob_high_mag": round(prob_high, 4),
        "historical_mean": c_info.get("mean_mag", 3.88),
        "historical_max": c_info.get("max_mag", 7.0),
        "historical_high_rate": c_info.get("high_mag_ratio", 0.5),
        "local_mae": c_info.get("mae", 0.28)
    }

def get_magnitude_tier(mag: float):
    """
    Returns standardized Richter magnitude severity class, color token,
    and descriptive impact criteria.
    """
    m = float(mag)
    if m < 3.0:
        return {
            "tier": "Micro",
            "badge_color": "cyan",
            "desc": "Imperceptible to humans; continuously detected and logged by sensitive seismographs.",
            "color_hex": "#06b6d4"
        }
    elif m < 4.0:
        return {
            "tier": "Minor",
            "badge_color": "emerald",
            "desc": "Often felt indoors by people at rest; rarely causes architectural or structural damage.",
            "color_hex": "#10b981"
        }
    elif m < 5.0:
        return {
            "tier": "Light",
            "badge_color": "amber",
            "desc": "Noticeable shaking of indoor items, rattling glassware. Minimal architectural disruption.",
            "color_hex": "#f59e0b"
        }
    elif m < 6.0:
        return {
            "tier": "Moderate",
            "badge_color": "amber",
            "desc": "Damage possible to poorly constructed buildings; slight damage to well-engineered structures.",
            "color_hex": "#fb923c"
        }
    elif m < 7.0:
        return {
            "tier": "Strong",
            "badge_color": "rose",
            "desc": "Destructive within an approximate 100-160 km radius in populated epicentral areas.",
            "color_hex": "#f43f5e"
        }
    elif m < 8.0:
        return {
            "tier": "Major",
            "badge_color": "rose",
            "desc": "Severe catastrophe; widespread destruction, structural collapses, and fault ruptures.",
            "color_hex": "#e11d48"
        }
    else:
        return {
            "tier": "Great",
            "badge_color": "rose",
            "desc": "Mega-thrust event capable of basin-wide destruction and trans-oceanic tsunami displacement.",
            "color_hex": "#9f1239"
        }

def calculate_felt_radius(magnitude: float, depth: float):
    """
    Approximates the felt radius (km) and epicentral Peak Ground Acceleration (%g).
    """
    m = float(magnitude)
    d = max(1.0, float(depth))
    felt_km = int(np.clip(10 ** (0.43 * m - 0.25), 5, 2500))
    log_pga = 0.26 * m - 0.008 * d - 0.95
    pga_pct_g = float(np.clip(10 ** log_pga, 0.01, 150.0))
    return {"felt_km": felt_km, "pga_pct_g": pga_pct_g}

def calculate_seismic_energy(magnitude: float):
    """
    Computes radiated seismic energy using the Gutenberg-Richter formula:
    log10(E) = 4.8 + 1.5 * M (E in Joules)
    Returns energy in Joules, Kilotons of TNT, and Megatons of TNT.
    """
    joules = 10 ** (4.8 + 1.5 * float(magnitude))
    tons_tnt = joules / 4.184e9
    kt_tnt = tons_tnt / 1e3
    mt_tnt = tons_tnt / 1e6
    return {
        "joules": joules,
        "tons_tnt": tons_tnt,
        "kt_tnt": kt_tnt,
        "mt_tnt": mt_tnt
    }

def get_mercalli_intensity(magnitude: float, depth: float):
    """
    Estimates Modified Mercalli Intensity (MMI) near the epicenter
    taking into account magnitude and focal depth attenuation.
    """
    mag = float(magnitude)
    d = max(1.0, float(depth))
    
    raw_mmi = 1.5 * mag - 1.5 * np.log10(d) + 1.2
    mmi = int(np.clip(np.round(raw_mmi), 1, 12))

    mmi_scale = {
        1: ("I", "Not Felt", "Imperceptible except by very few under especially favorable conditions."),
        2: ("II", "Weak", "Felt only by a few persons at rest, especially on upper floors."),
        3: ("III", "Weak", "Felt noticeably indoors, like the passing of a light truck."),
        4: ("IV", "Light", "Felt indoors by many, outdoors by few. Dishes, doors, windows disturbed."),
        5: ("V", "Moderate", "Felt by nearly everyone; many awakened. Some dishes and glassware broken."),
        6: ("VI", "Strong", "Felt by all, many frightened. Some heavy furniture moved; slight damage."),
        7: ("VII", "Very Strong", "Damage negligible in buildings of good design; slight to moderate in ordinary structures."),
        8: ("VIII", "Severe", "Damage slight in specially designed structures; considerable in ordinary buildings with partial collapse."),
        9: ("IX", "Violent", "Damage considerable in specially designed structures; buildings shifted off foundations. Ground cracked."),
        10: ("X", "Extreme", "Some well-built wooden structures destroyed; most masonry and frame structures destroyed. Landslides."),
        11: ("XI", "Extreme", "Few, if any, structures remain standing. Bridges destroyed. Broad fissures in ground."),
        12: ("XII", "Cataclysmic", "Total damage. Waves seen on ground surfaces. Objects thrown upward into the air.")
    }

    roman, label, desc = mmi_scale.get(mmi, ("V", "Moderate", "Noticeable vibration."))
    return {"roman": roman, "label": label, "description": desc, "raw_score": mmi}

def assess_tsunami_risk(magnitude: float, depth: float, latitude: float, longitude: float):
    """
    Evaluates potential oceanic tsunami hazard based on magnitude, depth,
    and seismic focus location.
    """
    mag = float(magnitude)
    d = float(depth)
    
    if mag < 4.5:
        return {
            "level": "Negligible",
            "color": "emerald",
            "text": "Magnitude below tsunami generation threshold. No oceanic perturbation anticipated."
        }
    elif mag < 6.5:
        return {
            "level": "Low",
            "color": "cyan",
            "text": "Small localized sea disturbances possible in enclosed bays if submarine rupture occurs."
        }
    elif mag < 7.5:
        if d <= 50.0:
            return {
                "level": "Elevated / Watch",
                "color": "amber",
                "text": "Shallow submarine focus capable of destructive local tsunamis within 100-300 km of epicenter."
            }
        else:
            return {
                "level": "Moderate",
                "color": "amber",
                "text": "Intermediate/deep focal depth attenuates seafloor vertical displacement. Localized wave risk."
            }
    else:
        if d <= 60.0:
            return {
                "level": "CRITICAL / WARNING",
                "color": "rose",
                "text": "Major submarine megathrust. Severe threat of ocean-wide destructive tsunami waves."
            }
        else:
            return {
                "level": "High Threat",
                "color": "rose",
                "text": "Severe magnitude event. Tsunami risk present for nearby coastal zones."
            }

def run_clt_simulation(population_series, sample_size: int, repetitions: int = 1000):
    """
    Vectorized Monte Carlo Central Limit Theorem simulator.
    """
    arr = population_series.dropna().to_numpy()
    pop_mean = float(arr.mean())
    pop_std = float(arr.std(ddof=1))

    idx = np.random.randint(0, len(arr), size=(repetitions, sample_size))
    samples = arr[idx]
    sample_means = samples.mean(axis=1)

    obs_mean = float(sample_means.mean())
    obs_se = float(sample_means.std(ddof=1))
    theo_se = float(pop_std / np.sqrt(sample_size))

    return {
        "sample_means": sample_means,
        "pop_mean": pop_mean,
        "pop_std": pop_std,
        "obs_mean": obs_mean,
        "obs_se": obs_se,
        "theo_se": theo_se
    }

def get_recent_live_events(df, limit: int = 25):
    """
    Extracts the latest seismic events recorded in the database.
    """
    if df is None or df.empty:
        return []
    sorted_df = df.sort_values("time", ascending=False).head(limit)
    return sorted_df[[
        "time", "place", "mag", "depth", "latitude", "longitude", "depth_category", "magnitude_category"
    ]].to_dict(orient="records")
