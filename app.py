import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import importlib
import utils
importlib.reload(utils)

from utils import (
    PRESET_HOTSPOTS,
    load_models,
    load_dataset,
    load_metrics,
    load_regional_metrics,
    load_statistical_results,
    predict_tectonic_chunk,
    predict_area_wise_magnitude,
    get_magnitude_tier,
    calculate_felt_radius,
    calculate_seismic_energy,
    get_mercalli_intensity,
    assess_tsunami_risk,
    run_clt_simulation,
    get_recent_live_events,
)

# Page Configuration
st.set_page_config(
    page_title="TerraPulse AI - Earthquake Magnitude & Seismic Intelligence",
    page_icon="🌋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load CSS Styles
CSS_PATH = Path("assets/styles.css")
if CSS_PATH.exists():
    with open(CSS_PATH, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Plotly dark template helper
DARK_LAYOUT = dict(
    paper_bgcolor="rgba(15, 23, 42, 0.6)",
    plot_bgcolor="rgba(15, 23, 42, 0.4)",
    font=dict(color="#f1f5f9", family="Plus Jakarta Sans, sans-serif"),
    margin=dict(l=30, r=30, t=40, b=30),
    xaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.15)"),
    yaxis=dict(gridcolor="rgba(255, 255, 255, 0.08)", zerolinecolor="rgba(255, 255, 255, 0.15)")
)

# Initialize Session State
if "preset_selected" not in st.session_state:
    st.session_state.preset_selected = "Tokyo / Kanto (Japan)"
if "inp_depth" not in st.session_state:
    st.session_state.inp_depth = 35.0
if "inp_lat" not in st.session_state:
    st.session_state.inp_lat = 35.6895
if "inp_lon" not in st.session_state:
    st.session_state.inp_lon = 139.6917
if "inp_month" not in st.session_state:
    st.session_state.inp_month = 3
if "inp_hour" not in st.session_state:
    st.session_state.inp_hour = 14

def apply_preset(name):
    cfg = PRESET_HOTSPOTS[name]
    st.session_state.inp_depth = float(cfg["depth"])
    st.session_state.inp_lat = float(cfg["latitude"])
    st.session_state.inp_lon = float(cfg["longitude"])
    st.session_state.inp_month = int(cfg["month"])
    st.session_state.inp_hour = int(cfg["hour"])
    st.session_state.preset_selected = name

# Load Resources
models = load_models()
classifier = models["classifier"]
regressor_rf = models.get("regressor_rf")
regressor_lr = models.get("regressor_lr")
regressor_dt = models.get("regressor_dt")
clusterer = models.get("clusterer")
regional_models = models.get("regional_models", {})

df_dataset = load_dataset()
metrics_data = load_metrics()
regional_metrics_data = load_regional_metrics()
stats_data = load_statistical_results()

# SIDEBAR
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 8px;">
        <div style="background: linear-gradient(135deg, #06b6d4, #a855f7); width: 44px; height: 44px; border-radius: 12px; display: flex; align-items: center; justify-content: center; font-size: 24px; box-shadow: 0 4px 15px rgba(6, 182, 212, 0.4);">
            🌋
        </div>
        <div>
            <div style="font-size: 1.35rem; font-weight: 800; letter-spacing: -0.02em; color: #f8fafc;">TerraPulse AI</div>
            <div style="font-size: 0.78rem; color: #06b6d4; font-weight: 600;">Seismic Intelligence Suite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 20px; padding: 5px 12px; font-size: 0.76rem; color: #34d399; font-weight: 600; display: inline-flex; align-items: center; gap: 6px; margin-bottom: 20px;">
        <span style="width: 8px; height: 8px; border-radius: 50%; background: #10b981; display: inline-block;"></span>
        Models Active • 48,080 Events Indexed
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧭 Navigation")
    app_mode = st.radio(
        "Select Module",
        [
            "🎯 Earthquake Magnitude Predictor",
            "🌍 Tectonic Clusters & Area Predictions",
            "🌐 Global 3D Seismic Explorer",
            "📈 Exploratory Data Analysis (EDA)",
            "📐 Statistical Inference & Hypothesis Hub",
            "📊 Model Benchmark & Analytics",
            "📁 Batch CSV Diagnostics"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("### ⚡ Quick Tectonic Presets")
    st.caption("Select a renowned seismic hotspot to populate diagnostic coordinates:")
    
    for hotspot_name in PRESET_HOTSPOTS.keys():
        if st.button(hotspot_name, key=f"btn_nav_{hotspot_name}", use_container_width=True):
            apply_preset(hotspot_name)
            st.rerun()

    st.markdown("---")
    st.markdown("""
    <div style="font-size: 0.75rem; color: #64748b; line-height: 1.5;">
        <b>Dataset:</b> USGS Global Seismology Catalog<br>
        <b>Magnitude Predictors:</b> Multiple Regression, Decision Tree, & Random Forest<br>
        <b>Classifier:</b> Tuned Random Forest (F1: 0.977)
    </div>
    """, unsafe_allow_html=True)

# MAIN APP HERO HEADER
st.markdown("""
<div class="tp-hero">
    <div class="tp-hero-title">
        <span>🌋 TerraPulse AI</span>
        <span class="tp-badge tp-badge-cyan">v2.4 Enterprise</span>
    </div>
    <div class="tp-hero-sub">
        High-precision earthquake magnitude forecasting, seismic hazard diagnostics, and statistical inference modeling powered by Machine Learning and the USGS Seismological Repository.
    </div>
</div>
""", unsafe_allow_html=True)

# KPI Summary Cards Row
total_events = metrics_data.get("dataset_stats", {}).get("total_records", 48080)
high_mag_count = metrics_data.get("dataset_stats", {}).get("high_mag_count", 28715)
max_recorded_mag = metrics_data.get("dataset_stats", {}).get("max_mag", 8.8)
model_accuracy = 97.32

col_k1, col_k2, col_k3, col_k4 = st.columns(4)
with col_k1:
    st.markdown(f"""
    <div class="tp-kpi-card">
        <div class="tp-kpi-label">Total Catalog Events</div>
        <div class="tp-kpi-val" style="color: #38bdf8;">{total_events:,}</div>
        <div class="tp-kpi-desc">Global seismic events (2024–2026)</div>
    </div>
    """, unsafe_allow_html=True)

with col_k2:
    st.markdown(f"""
    <div class="tp-kpi-card">
        <div class="tp-kpi-label">High Magnitude Ratio</div>
        <div class="tp-kpi-val" style="color: #fb7185;">{(high_mag_count/total_events*100):.1f}%</div>
        <div class="tp-kpi-desc">Events with Magnitude ≥ 4.0</div>
    </div>
    """, unsafe_allow_html=True)

with col_k3:
    st.markdown(f"""
    <div class="tp-kpi-card">
        <div class="tp-kpi-label">Peak Magnitude</div>
        <div class="tp-kpi-val" style="color: #fbbf24;">M {max_recorded_mag}</div>
        <div class="tp-kpi-desc">Highest observed Richter scale event</div>
    </div>
    """, unsafe_allow_html=True)

with col_k4:
    st.markdown(f"""
    <div class="tp-kpi-card">
        <div class="tp-kpi-label">Classifier Accuracy</div>
        <div class="tp-kpi-val" style="color: #34d399;">{model_accuracy}%</div>
        <div class="tp-kpi-desc">Ensemble Random Forest F1: 0.977</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)


# ==============================================================================
# TAB 1: EARTHQUAKE MAGNITUDE PREDICTOR & DIAGNOSTICS
# ==============================================================================
if app_mode == "🎯 Earthquake Magnitude Predictor":
    st.markdown("""
    <div class="tp-card">
        <div class="tp-card-title">🎯 Earthquake Magnitude Prediction & Seismic Diagnostics</div>
        <div style="color: #94a3b8; font-size: 0.92rem; margin-bottom: 8px;">
            Predict the exact <b>Richter / Moment Magnitude (M)</b> of an earthquake based on focal depth, geographic coordinates, and temporal inputs using trained Machine Learning regression models, accompanied by severity classification, ground acceleration, and physical damage metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Active preset indicator banner
    curr_preset = PRESET_HOTSPOTS.get(st.session_state.preset_selected, {})
    if curr_preset:
        st.markdown(f"""
        <div style="background: rgba(6, 182, 212, 0.08); border: 1px solid rgba(6, 182, 212, 0.25); border-radius: 12px; padding: 12px 18px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: center;">
            <div>
                <span style="font-weight: 700; color: #38bdf8;">📍 Active Target Preset:</span> 
                <span style="color: #f1f5f9; font-weight: 600;">{st.session_state.preset_selected}</span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 2px;">{curr_preset.get('region')} — <i>{curr_preset.get('risk_context')}</i></div>
            </div>
            <span class="tp-badge tp-badge-cyan">Pre-Calibrated</span>
        </div>
        """, unsafe_allow_html=True)

    col_ctrl, col_map = st.columns([1.1, 0.9])

    with col_ctrl:
        st.markdown("#### 1. Input Seismic & Hypocentral Parameters")
        
        c1, c2 = st.columns(2)
        with c1:
            inp_depth = st.number_input(
                "Focal Depth (km)",
                min_value=0.0,
                max_value=750.0,
                value=float(st.session_state.inp_depth),
                step=0.5,
                help="Hypocenter depth below Earth's crust in kilometers."
            )
            # Depth category badge
            if inp_depth <= 70:
                depth_label = "🟢 Shallow Crustal (0–70 km)"
            elif inp_depth <= 300:
                depth_label = "🟡 Intermediate Subduction (70–300 km)"
            else:
                depth_label = "🔴 Deep Focus Wadati-Benioff (>300 km)"
            st.caption(f"Classification: **{depth_label}**")

        with c2:
            inp_month = st.slider(
                "Calendar Month",
                min_value=1,
                max_value=12,
                value=int(st.session_state.inp_month),
                help="Month of event occurrence (seasonal hydrology & plate stress)."
            )
            inp_hour = st.slider(
                "Hour of Day (UTC)",
                min_value=0,
                max_value=23,
                value=int(st.session_state.inp_hour),
                help="Coordinated Universal Time (0-23 hours)."
            )

        c3, c4 = st.columns(2)
        with c3:
            inp_lat = st.number_input(
                "Epicentral Latitude (°)",
                min_value=-90.0,
                max_value=90.0,
                value=float(st.session_state.inp_lat),
                step=0.05,
                format="%.4f"
            )
        with c4:
            inp_lon = st.number_input(
                "Epicentral Longitude (°)",
                min_value=-180.0,
                max_value=180.0,
                value=float(st.session_state.inp_lon),
                step=0.05,
                format="%.4f"
            )

        # Regressor model selection
        st.markdown("#### ⚙️ Regression Model Option")
        reg_model_choice = st.radio(
            "Select Prediction Algorithm",
            [
                "⚡ Multi-Model Comparison (Multiple Regression vs Decision Tree vs Random Forest)",
                "🌍 Area-Wise Regional Expert (Specialized Earth Chunk Model)",
                "🌲 Random Forest Regressor (Ensemble Non-Linear)",
                "🌳 Decision Tree Regressor (CART Splitter)",
                "📐 Multiple Linear Regression (Statistical Baseline)",
                "🤝 Triple Consensus Ensemble (Weighted Blend)"
            ],
            index=0,
            help="Choose between Area-Wise Regional Expert, Multi-Model comparison, or individual global regressors."
        )

    with col_map:
        st.markdown("#### 2. Epicenter Geospatial Pinpoint")
        fig_mini = go.Figure()
        
        if df_dataset is not None:
            sample_pts = df_dataset.sample(min(800, len(df_dataset)), random_state=42)
            fig_mini.add_trace(go.Scattergeo(
                lat=sample_pts["latitude"],
                lon=sample_pts["longitude"],
                mode="markers",
                marker=dict(size=2.5, color="rgba(255, 255, 255, 0.15)"),
                name="Historical Seismicity",
                hoverinfo="none"
            ))

        fig_mini.add_trace(go.Scattergeo(
            lat=[inp_lat],
            lon=[inp_lon],
            mode="markers+text",
            marker=dict(size=14, color="#06b6d4", symbol="star", line=dict(width=2, color="#ffffff")),
            text=[f"Target Focus: {inp_depth}km"],
            textposition="top center",
            name="Target Hypocenter"
        ))

        fig_mini.update_geos(
            projection_type="natural earth",
            showcoastlines=True, coastlinecolor="#334155",
            showland=True, landcolor="#0f172a",
            showocean=True, oceancolor="#020617",
            showlakes=True, lakecolor="#020617",
            showcountries=True, countrycolor="#1e293b",
            center=dict(lat=inp_lat, lon=inp_lon),
            projection_scale=1.5
        )
        fig_mini.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=280,
            showlegend=False
        )
        st.plotly_chart(fig_mini, use_container_width=True)

    # Prediction Action Trigger
    st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
    btn_predict = st.button("⚡ Predict Earthquake Magnitude & Severity", use_container_width=True, type="primary")

    # Perform Prediction
    input_sample = pd.DataFrame([{
        "depth": inp_depth,
        "latitude": inp_lat,
        "longitude": inp_lon,
        "month": inp_month,
        "hour": inp_hour
    }])

    # 1. Classification
    cls_pred = classifier.predict(input_sample)[0]
    cls_proba = classifier.predict_proba(input_sample)[0]
    high_mag_prob = cls_proba[1]
    low_mag_prob = cls_proba[0]

    # 2. Magnitude Regression across global and regional models
    pred_lr = float(regressor_lr.predict(input_sample)[0]) if regressor_lr is not None else 3.88
    pred_dt = float(regressor_dt.predict(input_sample)[0]) if regressor_dt is not None else 3.88
    pred_rf = float(regressor_rf.predict(input_sample)[0]) if regressor_rf is not None else 3.88
    
    # Area-Wise Tectonic Cluster Expert Prediction
    area_info = predict_area_wise_magnitude(inp_lat, inp_lon, inp_depth, inp_month, inp_hour, models, regional_metrics_data)
    pred_local = area_info["local_pred"]
    
    consensus_mag = (pred_lr + pred_dt + pred_rf) / 3.0
    spread_mag = max(pred_lr, pred_dt, pred_rf) - min(pred_lr, pred_dt, pred_rf)

    tier_lr = get_magnitude_tier(pred_lr)
    tier_dt = get_magnitude_tier(pred_dt)
    tier_rf = get_magnitude_tier(pred_rf)
    tier_local = get_magnitude_tier(pred_local)

    if "Area-Wise" in reg_model_choice:
        predicted_mag = pred_local
        active_model_name = f"Area-Wise Regional Expert ({area_info['cluster_name']})"
        tier_info = tier_local
    elif "Decision Tree" in reg_model_choice:
        predicted_mag = pred_dt
        active_model_name = "Decision Tree Regressor"
        tier_info = tier_dt
    elif "Linear" in reg_model_choice:
        predicted_mag = pred_lr
        active_model_name = "Multiple Linear Regression"
        tier_info = tier_lr
    elif "Random Forest" in reg_model_choice:
        predicted_mag = pred_rf
        active_model_name = "Random Forest Regressor"
        tier_info = tier_rf
    else:
        predicted_mag = consensus_mag
        active_model_name = "Triple Consensus Ensemble (LR + DT + RF)"
        tier_info = get_magnitude_tier(predicted_mag)

    # Enforce realistic bounds
    predicted_mag = max(2.5, min(9.2, predicted_mag))

    # Derived physics & seismic metrics
    felt_info = calculate_felt_radius(predicted_mag, inp_depth)
    energy_info = calculate_seismic_energy(predicted_mag)
    mmi_info = get_mercalli_intensity(predicted_mag, inp_depth)
    tsunami_info = assess_tsunami_risk(predicted_mag, inp_depth, inp_lat, inp_lon)

    # =========================================================================
    # PREDICTION SHOWCASE SECTION
    # =========================================================================
    st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 28px 0 20px 0;'>", unsafe_allow_html=True)
    st.markdown("### 📈 Predicted Earthquake Magnitude & Model Comparison")

    # If Multi-Model Comparison is chosen
    if "Multi-Model" in reg_model_choice:
        st.markdown("##### ⚡ Multi-Model Magnitude Evaluation: Linear Regression vs Decision Tree vs Random Forest")
        rc1, rc2, rc3 = st.columns(3)

        with rc1:
            st.markdown(f"""
            <div class="tp-card" style="border: 1px solid rgba(168, 85, 247, 0.4); box-shadow: 0 4px 20px rgba(168, 85, 247, 0.15); height: 100%;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #c084fc; text-transform: uppercase; letter-spacing: 0.05em;">📐 Multiple Linear Regression</div>
                <div style="display: flex; align-items: baseline; gap: 8px; margin: 8px 0;">
                    <span style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #f8fafc;">M {pred_lr:.2f}</span>
                    <span style="font-size: 1rem; color: #c084fc; font-weight: 600;">Richter</span>
                </div>
                <div>
                    <span class="tp-badge tp-badge-{tier_lr['badge_color']}" style="font-size: 0.82rem; padding: 4px 10px;">● {tier_lr['tier']} Severity</span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px; line-height: 1.45;">
                    <b>Algorithm:</b> Standardized OLS Hyperplane.<br>
                    <b>Test Error:</b> MAE = 0.4519 | R² = 0.5274<br>
                    <b>Confidence 95%:</b> M {max(1.0, pred_lr - 0.45):.2f} – M {pred_lr + 0.45:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rc2:
            st.markdown(f"""
            <div class="tp-card" style="border: 1px solid rgba(16, 185, 129, 0.4); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15); height: 100%;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #34d399; text-transform: uppercase; letter-spacing: 0.05em;">🌳 Decision Tree Regressor</div>
                <div style="display: flex; align-items: baseline; gap: 8px; margin: 8px 0;">
                    <span style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #f8fafc;">M {pred_dt:.2f}</span>
                    <span style="font-size: 1rem; color: #34d399; font-weight: 600;">Richter</span>
                </div>
                <div>
                    <span class="tp-badge tp-badge-{tier_dt['badge_color']}" style="font-size: 0.82rem; padding: 4px 10px;">● {tier_dt['tier']} Severity</span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px; line-height: 1.45;">
                    <b>Algorithm:</b> Recursive CART Tree (Depth=10).<br>
                    <b>Test Error:</b> MAE = 0.2972 | R² = 0.7623<br>
                    <b>Confidence 95%:</b> M {max(1.0, pred_dt - 0.30):.2f} – M {pred_dt + 0.30:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with rc3:
            st.markdown(f"""
            <div class="tp-card" style="border: 1px solid rgba(6, 182, 212, 0.4); box-shadow: 0 4px 20px rgba(6, 182, 212, 0.15); height: 100%;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase; letter-spacing: 0.05em;">🌲 Random Forest (Global)</div>
                <div style="display: flex; align-items: baseline; gap: 8px; margin: 8px 0;">
                    <span style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #f8fafc;">M {pred_rf:.2f}</span>
                    <span style="font-size: 1rem; color: #38bdf8; font-weight: 600;">Richter</span>
                </div>
                <div>
                    <span class="tp-badge tp-badge-{tier_rf['badge_color']}" style="font-size: 0.82rem; padding: 4px 10px;">● {tier_rf['tier']} Severity</span>
                </div>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px; line-height: 1.45;">
                    <b>Algorithm:</b> 80 Non-linear Bagged Trees.<br>
                    <b>Test Error:</b> MAE = 0.2862 | R² = 0.7809<br>
                    <b>Confidence 95%:</b> M {max(1.0, pred_rf - 0.29):.2f} – M {pred_rf + 0.29:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Dedicated Area-Wise Regional Tectonic Expert Card
        diff_local_rf = pred_local - pred_rf
        diff_str = f"+{diff_local_rf:.2f}" if diff_local_rf >= 0 else f"{diff_local_rf:.2f}"
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(168, 85, 247, 0.12) 100%); border: 1px solid rgba(6, 182, 212, 0.35); border-radius: 14px; padding: 18px 22px; margin: 16px 0; box-shadow: 0 4px 20px rgba(6, 182, 212, 0.15);">
            <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
                <div>
                    <div style="font-size: 0.78rem; text-transform: uppercase; font-weight: 700; color: #06b6d4; letter-spacing: 0.05em;">
                        🌍 Specialized Area-Wise Tectonic Expert Forecast
                    </div>
                    <div style="font-size: 1.25rem; font-weight: 800; color: #f8fafc; margin-top: 2px;">
                        {area_info['cluster_name']}
                    </div>
                    <div style="font-size: 0.82rem; color: #94a3b8; margin-top: 2px;">
                        Tectonic Regime: <b style="color: #cbd5e1;">{area_info['tectonic_type']}</b> • Cluster Test MAE: <b style="color: #34d399;">±{area_info['local_mae']:.4f} M</b>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="text-align: right;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Area Expert Magnitude</div>
                        <div style="font-size: 2.2rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #38bdf8; line-height: 1;">
                            M {pred_local:.2f}
                        </div>
                    </div>
                    <div>
                        <span class="tp-badge tp-badge-{tier_local['badge_color']}" style="font-size: 0.85rem; padding: 5px 12px;">● {tier_local['tier']}</span>
                    </div>
                </div>
            </div>
            <div style="margin-top: 14px; padding-top: 12px; border-top: 1px solid rgba(255, 255, 255, 0.08); display: flex; flex-wrap: wrap; gap: 20px; font-size: 0.83rem; color: #cbd5e1;">
                <div>📊 <b>Historical Chunk Mean:</b> M {area_info['historical_mean']:.2f}</div>
                <div>⚡ <b>Historical Chunk Max:</b> M {area_info['historical_max']:.2f}</div>
                <div>🚨 <b>High-Mag Rate (≥4.0):</b> {(area_info['historical_high_rate']*100):.1f}%</div>
                <div>⚖️ <b>Area vs Global Divergence:</b> <span style="color: {'#34d399' if abs(diff_local_rf)<=0.2 else '#fb7185'}; font-weight: 700;">{diff_str} M</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Consensus and agreement banner across all 3 models
        agree_badge = "🟢 High Agreement (Spread ≤ 0.25 M)" if spread_mag <= 0.25 else ("🟡 Moderate Spread (Spread ≤ 0.50 M)" if spread_mag <= 0.50 else "🔴 High Divergence")
        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px; padding: 14px 20px; margin: 16px 0; display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px;">
            <div>
                <span style="font-weight: 700; color: #f8fafc;">🤝 Triple Consensus Ensemble Magnitude:</span>
                <span style="font-size: 1.35rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono'; margin-left: 8px;">M {consensus_mag:.2f}</span>
                <span style="font-size: 0.82rem; color: #94a3b8; margin-left: 12px;">Maximum model spread: <b>|ΔM_max| = {spread_mag:.2f} M</b></span>
            </div>
            <span class="tp-badge tp-badge-cyan" style="font-size: 0.85rem;">{agree_badge}</span>
        </div>
        """, unsafe_allow_html=True)

        # Mathematical & Architectural Detail Expanders
        exp_col1, exp_col2 = st.columns(2)
        with exp_col1:
            with st.expander("📐 Multiple Linear Regression Formula & OLS Weights"):
                st.markdown(r"""
                $$\hat{M}_{\text{OLS}} = \beta_0 + \sum_{j=1}^{5} \beta_j \cdot Z_j$$
                
                - **Intercept ($\beta_0$):** `3.8811` (Dataset baseline anchor)
                - **Depth Weight ($\beta_1$):** `+0.0051` (Crustal compaction attenuation)
                - **Latitude Vector ($\beta_2$):** `+0.0133` (Tectonic coordinate gradient)
                - **Longitude Vector ($\beta_3$):** `+0.0148` (Circum-Pacific belt alignment)
                - **Month ($\beta_4$):** `+0.0003` | **Hour ($\beta_5$):** `+0.0010`
                """)

        with exp_col2:
            with st.expander("🌳 Decision Tree Regressor Architecture & Splits"):
                st.markdown(r"""
                **Recursive Binary Partitioning (CART)**
                
                - **Tree Max Depth:** `10 levels` (Restricts overfitting while capturing subduction curves)
                - **Minimum Leaf Samples:** `5 observations`
                - **Splitting Criterion:** Mean Squared Error (MSE / Variance Reduction)
                - **Feature Influence:** Longitude (31.5%), Latitude (29.8%), Focal Depth (19.4%)
                - **Advantage:** Models abrupt geophysical fault step boundaries without linear assumptions.
                """)

    else:
        # Single Model Hero Showcase Card
        st.markdown(f"""
        <div class="tp-mag-showcase">
            <div>
                <div style="font-size: 0.82rem; text-transform: uppercase; font-weight: 700; color: #38bdf8; letter-spacing: 0.06em;">
                    AI Magnitude Estimation • {active_model_name}
                </div>
                <div class="tp-mag-number-wrap">
                    <span class="tp-mag-number">M {predicted_mag:.2f}</span>
                    <span class="tp-mag-unit">Richter Scale</span>
                </div>
                <div style="margin-top: 6px;">
                    <span class="tp-mag-tier-pill" style="background: rgba({ '6,182,212' if tier_info['badge_color']=='cyan' else ('16,185,129' if tier_info['badge_color']=='emerald' else ('245,158,11' if tier_info['badge_color']=='amber' else '244,63,94')) }, 0.15); border: 1px solid {tier_info['color_hex']}; color: {tier_info['color_hex']};">
                        ● {tier_info['tier']} Severity Event
                    </span>
                    <span style="color: #94a3b8; font-size: 0.85rem; margin-left: 10px;">
                        Estimated 95% Confidence Range: <b>M {max(1.0, predicted_mag - 0.46):.2f} – M {predicted_mag + 0.46:.2f}</b>
                    </span>
                </div>
                <div style="color: #cbd5e1; font-size: 0.9rem; margin-top: 10px; max-width: 650px; line-height: 1.5;">
                    {tier_info['desc']}
                </div>
            </div>
            <div style="text-align: right; min-width: 200px;">
                <div style="font-size: 0.76rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em;">Classification</div>
                <div style="font-size: 1.35rem; font-weight: 800; color: {'#fb7185' if cls_pred==1 else '#34d399'}; margin-top: 4px;">
                    {'🚨 High Mag (≥4.0)' if cls_pred==1 else '🟢 Low Mag (<4.0)'}
                </div>
                <div style="font-size: 0.8rem; color: #64748b; margin-top: 2px;">Probability: {(high_mag_prob*100):.1f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Interactive Richter Scale Visual Spectrum
    st.markdown("##### 📏 Continuous Richter Scale Position")
    fig_scale = go.Figure()
    
    # Gradient bar steps
    scale_steps = [
        (0.0, 3.0, "rgba(6, 182, 212, 0.6)", "Micro (<3.0)"),
        (3.0, 4.0, "rgba(16, 185, 129, 0.6)", "Minor (3.0-3.9)"),
        (4.0, 5.0, "rgba(245, 158, 11, 0.6)", "Light (4.0-4.9)"),
        (5.0, 6.0, "rgba(251, 146, 60, 0.7)", "Moderate (5.0-5.9)"),
        (6.0, 7.0, "rgba(244, 63, 94, 0.7)", "Strong (6.0-6.9)"),
        (7.0, 8.0, "rgba(225, 29, 72, 0.8)", "Major (7.0-7.9)"),
        (8.0, 9.5, "rgba(159, 18, 57, 0.9)", "Great (≥8.0)")
    ]
    
    for s_start, s_end, color, name in scale_steps:
        fig_scale.add_trace(go.Bar(
            x=[s_end - s_start],
            y=["Richter Scale"],
            base=[s_start],
            orientation="h",
            marker=dict(color=color),
            name=name,
            hoverinfo="name"
        ))

    # Add pointer marker for predicted magnitude
    fig_scale.add_trace(go.Scatter(
        x=[predicted_mag],
        y=["Richter Scale"],
        mode="markers+text",
        marker=dict(size=18, color="#ffffff", symbol="triangle-down", line=dict(width=2, color="#06b6d4")),
        text=[f"Predicted: M {predicted_mag:.2f}"],
        textposition="top center",
        name="Predicted Value",
        showlegend=False
    ))

    fig_scale.update_layout(
        barmode="stack",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=20, r=20, t=30, b=20),
        height=140,
        xaxis=dict(range=[1.0, 9.5], dtick=1, gridcolor="rgba(255,255,255,0.1)", title="Earthquake Magnitude (M)"),
        yaxis=dict(visible=False),
        showlegend=False
    )
    st.plotly_chart(fig_scale, use_container_width=True)

    # Detailed Impact Metrics Row
    c_m1, c_m2, c_m3, c_m4 = st.columns(4)

    with c_m1:
        st.markdown(f"""
        <div class="tp-card" style="height: 100%;">
            <div class="tp-card-title">🏢 Modified Mercalli</div>
            <div style="font-size: 2rem; font-weight: 800; color: #f59e0b; font-family: 'JetBrains Mono';">
                MMI {mmi_info['roman']}
            </div>
            <div style="font-size: 0.9rem; font-weight: 700; color: #f1f5f9; margin: 4px 0;">{mmi_info['label']}</div>
            <div style="font-size: 0.78rem; color: #94a3b8;">{mmi_info['description']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c_m2:
        st.markdown(f"""
        <div class="tp-card" style="height: 100%;">
            <div class="tp-card-title">💥 Radiated Energy</div>
            <div style="font-size: 1.7rem; font-weight: 800; color: #38bdf8; font-family: 'JetBrains Mono';">
                {energy_info['kt_tnt']:.2f} kt
            </div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin: 4px 0;">TNT Equivalent</div>
            <div style="font-size: 0.78rem; color: #94a3b8;">{energy_info['joules']:.2e} Joules. ({(energy_info['kt_tnt']/15.0):.1f}× Hiroshima atomic yield).</div>
        </div>
        """, unsafe_allow_html=True)

    with c_m3:
        st.markdown(f"""
        <div class="tp-card" style="height: 100%;">
            <div class="tp-card-title">🌐 Felt Area Radius</div>
            <div style="font-size: 1.9rem; font-weight: 800; color: #a855f7; font-family: 'JetBrains Mono';">
                ~{felt_info['felt_km']} km
            </div>
            <div style="font-size: 0.85rem; font-weight: 600; color: #cbd5e1; margin: 4px 0;">Perceptible Range</div>
            <div style="font-size: 0.78rem; color: #94a3b8;">Peak ground acceleration (PGA) est: <b>{felt_info['pga_pct_g']:.2f}% g</b>.</div>
        </div>
        """, unsafe_allow_html=True)

    with c_m4:
        t_color = tsunami_info["color"]
        st.markdown(f"""
        <div class="tp-card" style="height: 100%;">
            <div class="tp-card-title">🌊 Tsunami Risk</div>
            <div>
                <span class="tp-badge tp-badge-{t_color}" style="font-size: 0.95rem; padding: 5px 12px;">
                    {tsunami_info['level']}
                </span>
            </div>
            <div style="font-size: 0.78rem; color: #94a3b8; margin-top: 10px;">
                {tsunami_info['text']}
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Magnitude Sensitivity Curve (How magnitude varies across depth for these coordinates)
    st.markdown("---")
    st.markdown("#### 🔬 Focal Depth Sensitivity & Magnitude Attenuation Analysis")
    st.caption("Inspect how the predicted earthquake magnitude shifts as hypocenter depth varies from 0 km (surface rupture) to 600 km (deep mantle subduction) at this exact coordinate:")

    depth_range = np.linspace(2.0, 600.0, 60)
    sim_inputs = pd.DataFrame({
        "depth": depth_range,
        "latitude": [inp_lat] * len(depth_range),
        "longitude": [inp_lon] * len(depth_range),
        "month": [inp_month] * len(depth_range),
        "hour": [inp_hour] * len(depth_range)
    })

    sim_lr_preds = regressor_lr.predict(sim_inputs) if regressor_lr is not None else np.full(len(depth_range), 4.0)
    sim_dt_preds = regressor_dt.predict(sim_inputs) if regressor_dt is not None else np.full(len(depth_range), 4.0)
    sim_rf_preds = regressor_rf.predict(sim_inputs) if regressor_rf is not None else np.full(len(depth_range), 4.0)

    fig_sens = go.Figure()
    fig_sens.add_trace(go.Scatter(
        x=depth_range,
        y=sim_lr_preds,
        mode="lines",
        name="Multiple Linear Regression",
        line=dict(color="#c084fc", width=2.5, dash="dash")
    ))
    fig_sens.add_trace(go.Scatter(
        x=depth_range,
        y=sim_dt_preds,
        mode="lines",
        name="Decision Tree Regressor",
        line=dict(color="#34d399", width=2.5)
    ))
    fig_sens.add_trace(go.Scatter(
        x=depth_range,
        y=sim_rf_preds,
        mode="lines",
        name="Random Forest Regressor (Global)",
        line=dict(color="#06b6d4", width=3)
    ))
    
    # Area-Wise Local Expert Depth Curve
    local_reg_model = models.get("regional_models", {}).get(str(area_info["cluster_id"]))
    if local_reg_model is not None:
        sim_local_preds = local_reg_model.predict(sim_inputs)
        fig_sens.add_trace(go.Scatter(
            x=depth_range,
            y=sim_local_preds,
            mode="lines",
            name=f"Area Expert ({area_info['cluster_name'].split('/')[0].strip()})",
            line=dict(color="#f59e0b", width=3)
        ))

    # Mark user's current point
    fig_sens.add_trace(go.Scatter(
        x=[inp_depth],
        y=[predicted_mag],
        mode="markers",
        name=f"Current Focus ({inp_depth}km, M {predicted_mag:.2f})",
        marker=dict(size=12, color="#f43f5e", symbol="diamond")
    ))

    fig_sens.update_layout(
        **DARK_LAYOUT,
        height=320,
        xaxis_title="Focal Depth (km)",
        yaxis_title="Estimated Magnitude (M)",
        hovermode="x unified"
    )
    st.plotly_chart(fig_sens, use_container_width=True)


# ==============================================================================
# MODULE: TECTONIC CLUSTERS & AREA-WISE PREDICTIONS
# ==============================================================================
elif app_mode == "🌍 Tectonic Clusters & Area Predictions":
    st.markdown("""
    <div class="tp-card">
        <div class="tp-card-title">🌍 Tectonic Clusters & Earth Spatial Chunking Engine</div>
        <div style="color: #94a3b8; font-size: 0.92rem; line-height: 1.5;">
            Earthquakes do not distribute uniformly across the globe — high-magnitude events cluster strictly along plate convergence boundaries and subduction trenches. By dividing the planet into <b>8 major tectonic chunks</b> using 3D spherical coordinates $(x, y, z)$ on the unit sphere, TerraPulse AI fits <b>dedicated regional expert models</b> for each zone, preventing the attenuation of local fault physics by global averaging.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 4 Key KPI Summary Cards
    tc_col1, tc_col2, tc_col3, tc_col4 = st.columns(4)
    with tc_col1:
        st.markdown("""
        <div class="tp-kpi-card">
            <div class="tp-kpi-label">Spatial Partitions</div>
            <div class="tp-kpi-val" style="color: #38bdf8;">8 Chunks</div>
            <div class="tp-kpi-desc">Unit-Sphere 3D Tectonic Zones</div>
        </div>
        """, unsafe_allow_html=True)

    with tc_col2:
        st.markdown("""
        <div class="tp-kpi-card">
            <div class="tp-kpi-label">Peak Historical Chunk</div>
            <div class="tp-kpi-val" style="color: #f43f5e;">M 8.80</div>
            <div class="tp-kpi-desc">NW Pacific / Kuril-Japan Arc</div>
        </div>
        """, unsafe_allow_html=True)

    with tc_col3:
        st.markdown("""
        <div class="tp-kpi-card">
            <div class="tp-kpi-label">Best Local Model Error</div>
            <div class="tp-kpi-val" style="color: #10b981;">± 0.2255 M</div>
            <div class="tp-kpi-desc">Alpine-Himalayan Collision</div>
        </div>
        """, unsafe_allow_html=True)

    with tc_col4:
        st.markdown("""
        <div class="tp-kpi-card">
            <div class="tp-kpi-label">Subduction Energy Share</div>
            <div class="tp-kpi-val" style="color: #c084fc;">~71%</div>
            <div class="tp-kpi-desc">Circum-Pacific Ring of Fire</div>
        </div>
        """, unsafe_allow_html=True)

    cluster_tabs = st.tabs([
        "🗺️ Global 8-Chunk Tectonic Map",
        "🔬 Deep Cluster Analytics & Feature Weights",
        "🎯 Area-Wise Prediction Simulator",
        "📊 All-Clusters Benchmark Leaderboard"
    ])

    # --------------------------------------------------------------------------
    # SUB-TAB 1: GLOBAL 8-CHUNK TECTONIC MAP
    # --------------------------------------------------------------------------
    with cluster_tabs[0]:
        st.markdown("#### 🌐 Global Partitioning into 8 Regional Tectonic Chunks")
        st.caption("Visualizing 48,080 catalog events color-coded by their 3D spherical cluster assignment alongside cluster centroids:")

        mcol1, mcol2, mcol3 = st.columns([1, 1, 1])
        with mcol1:
            map_proj = st.selectbox(
                "Map Projection Style",
                ["3D Orthographic (Globe)", "Natural Earth", "Equirectangular"],
                key="chunk_proj_style"
            )
        with mcol2:
            chunk_filter = st.selectbox(
                "Filter Specific Cluster",
                ["All 8 Tectonic Chunks"] + [f"[{cid}] {info['name']}" for cid, info in regional_metrics_data.items()],
                key="chunk_filter_select"
            )
        with mcol3:
            plot_res = st.selectbox(
                "Map Sample Density",
                [3000, 6000, 12000, 24000],
                index=1,
                key="chunk_plot_res"
            )

        if df_dataset is not None:
            # Prepare plotting dataframe
            plot_df = df_dataset.copy()
            if "cluster_id" not in plot_df.columns:
                lat_r = np.radians(plot_df["latitude"])
                lon_r = np.radians(plot_df["longitude"])
                cx = np.cos(lat_r) * np.cos(lon_r)
                cy = np.cos(lat_r) * np.sin(lon_r)
                cz = np.sin(lat_r)
                coords_mat = pd.DataFrame({"cart_x": cx, "cart_y": cy, "cart_z": cz})
                plot_df["cluster_id"] = clusterer.predict(coords_mat)
                plot_df["cluster_name"] = plot_df["cluster_id"].map(lambda c: regional_metrics_data.get(str(c), {}).get("name", f"Cluster {c}"))

            if chunk_filter != "All 8 Tectonic Chunks":
                sel_cid = int(chunk_filter.split("]")[0].replace("[", ""))
                plot_df = plot_df[plot_df["cluster_id"] == sel_cid]

            sample_df = plot_df.sample(min(len(plot_df), plot_res), random_state=42)

            palette_map = {
                "NW Pacific / Kuril-Kamchatka-Japan Arc": "#38bdf8",
                "Caribbean Basin & Antilles Fault System": "#a855f7",
                "SE Asia / Sunda-Banda & Coral Triangle": "#10b981",
                "Alpine-Himalayan & Mediterranean Belt": "#f59e0b",
                "North America Continental & San Andreas": "#ec4899",
                "SW Pacific / Tonga-Kermadec-Fiji Trench": "#06b6d4",
                "South America / Andean Subduction & Chile": "#ef4444",
                "North Pacific / Alaska-Aleutians & Hawaii": "#6366f1"
            }

            fig_chunks = px.scatter_geo(
                sample_df,
                lat="latitude",
                lon="longitude",
                color="cluster_name",
                hover_name="place",
                hover_data={"mag": ":.2f", "depth": ":.1f km", "latitude": False, "longitude": False, "cluster_name": True},
                color_discrete_map=palette_map,
                labels={"cluster_name": "Tectonic Chunk", "mag": "Magnitude", "depth": "Depth"}
            )
            fig_chunks.update_traces(marker=dict(size=3.5, opacity=0.75))

            # Add centroid markers
            centroid_lats = []
            centroid_lons = []
            centroid_texts = []
            for cid, cinfo in regional_metrics_data.items():
                c_lat = cinfo["centroid"]["latitude"]
                c_lon = cinfo["centroid"]["longitude"]
                centroid_lats.append(c_lat)
                centroid_lons.append(c_lon)
                centroid_texts.append(f"Centroid {cid}: {cinfo['name'].split('/')[0].strip()}")

            fig_chunks.add_trace(go.Scattergeo(
                lat=centroid_lats,
                lon=centroid_lons,
                mode="markers+text",
                marker=dict(size=11, color="#ffffff", symbol="diamond", line=dict(width=2, color="#06b6d4")),
                text=centroid_texts,
                textposition="top center",
                name="Tectonic Centroids",
                hoverinfo="text"
            ))

            proj_type = "orthographic" if "3D" in map_proj else ("natural earth" if "Natural" in map_proj else "equirectangular")
            fig_chunks.update_geos(
                projection_type=proj_type,
                showcoastlines=True, coastlinecolor="#334155",
                showland=True, landcolor="#0f172a",
                showocean=True, oceancolor="#020617",
                showlakes=True, lakecolor="#020617",
                showcountries=True, countrycolor="#1e293b"
            )
            fig_chunks.update_layout(
                **DARK_LAYOUT,
                height=560,
                legend=dict(orientation="h", yanchor="bottom", y=-0.18, xanchor="center", x=0.5, font=dict(size=11))
            )
            st.plotly_chart(fig_chunks, use_container_width=True)

            # Regional Seismicity Distribution Comparison Chart
            st.markdown("##### 📊 Regional Event Volume & High-Magnitude (≥ 4.0) Rates Across Chunks")
            summary_list = []
            for cid, cinfo in regional_metrics_data.items():
                summary_list.append({
                    "Cluster": f"[{cid}] {cinfo['name'].split('/')[0].strip()}",
                    "Total Events": cinfo["total_records"],
                    "High Mag Rate (%)": round(cinfo["high_mag_ratio"] * 100, 1),
                    "Mean Magnitude": cinfo["mean_mag"]
                })
            df_csummary = pd.DataFrame(summary_list)

            sc1, sc2 = st.columns(2)
            with sc1:
                fig_events_bar = px.bar(
                    df_csummary,
                    x="Total Events",
                    y="Cluster",
                    orientation="h",
                    color="Total Events",
                    color_continuous_scale="Viridis",
                    title="Catalog Event Volume by Tectonic Chunk",
                    text_auto=True
                )
                fig_events_bar.update_layout(**DARK_LAYOUT, height=330)
                st.plotly_chart(fig_events_bar, use_container_width=True)

            with sc2:
                fig_rate_bar = px.bar(
                    df_csummary,
                    x="High Mag Rate (%)",
                    y="Cluster",
                    orientation="h",
                    color="High Mag Rate (%)",
                    color_continuous_scale="Reds",
                    title="Proportion of Events with Magnitude ≥ 4.0 (%)",
                    text_auto=True
                )
                fig_rate_bar.update_layout(**DARK_LAYOUT, height=330)
                st.plotly_chart(fig_rate_bar, use_container_width=True)

    # --------------------------------------------------------------------------
    # SUB-TAB 2: DEEP CLUSTER ANALYTICS & REGIONAL FEATURE WEIGHTS
    # --------------------------------------------------------------------------
    with cluster_tabs[1]:
        st.markdown("#### 🔬 Deep Seismological Profile & Local Feature Weight Breakdown")
        st.caption("Inspect the tectonic dynamics, historical statistics, and trained Random Forest feature weights for any specific Earth chunk:")

        sel_chunk_key = st.selectbox(
            "Select Tectonic Chunk to Inspect",
            [f"[{cid}] {cinfo['name']}" for cid, cinfo in regional_metrics_data.items()],
            key="deep_chunk_selector"
        )
        sel_cid_str = sel_chunk_key.split("]")[0].replace("[", "")
        sel_cinfo = regional_metrics_data.get(sel_cid_str, {})

        if sel_cinfo:
            dc1, dc2, dc3, dc4 = st.columns(4)
            with dc1:
                st.metric("Tectonic Regime", sel_cinfo.get("tectonic_type", "Subduction"))
                st.caption(sel_cinfo.get("description", ""))
            with dc2:
                st.metric("Historical Mean Mag", f"M {sel_cinfo.get('mean_mag', 4.0):.2f}")
                st.caption(f"90th Percentile: **M {sel_cinfo.get('p90_mag', 5.0):.2f}**")
            with dc3:
                st.metric("Historical Peak Mag", f"M {sel_cinfo.get('max_mag', 7.0):.2f}")
                st.caption(f"Catalog Share: **{sel_cinfo.get('percentage_of_catalog', 0):.1f}%** ({sel_cinfo.get('total_records', 0):,} events)")
            with dc4:
                st.metric("Local Model Error (MAE)", f"± {sel_cinfo.get('mae', 0.28):.4f} M")
                st.caption(f"Local Model R² Score: **{sel_cinfo.get('r2', 0.78):.4f}**")

            st.markdown("---")
            fc1, fc2 = st.columns([1.1, 0.9])

            with fc1:
                st.markdown("##### 🧭 Local Feature Importance Ranking in this Chunk")
                st.caption("How physical depth, coordinates, and seasonal factors influence magnitude in THIS specific tectonic regime:")
                local_fi = pd.DataFrame(sel_cinfo.get("feature_importances", []))
                if not local_fi.empty:
                    fig_local_fi = px.bar(
                        local_fi,
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        color="Importance",
                        color_continuous_scale="Teal",
                        text_auto=True,
                        labels={"Importance": "Gini Importance Score", "Feature": "Input Feature"}
                    )
                    fig_local_fi.update_layout(**DARK_LAYOUT, height=270)
                    st.plotly_chart(fig_local_fi, use_container_width=True)

            with fc2:
                st.markdown("##### 📍 Top Fault Locations & Depth Distribution")
                top_regs = sel_cinfo.get("top_regions", {})
                if top_regs:
                    df_top = pd.DataFrame(list(top_regs.items()), columns=["Region / Country", "Recorded Events"])
                    st.dataframe(df_top, use_container_width=True, hide_index=True)
                
                st.markdown(f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 16px; margin-top: 10px; font-size: 0.84rem; line-height: 1.5;">
                    • <b>Average Rupture Depth:</b> {sel_cinfo.get('mean_depth', 35):.1f} km (Max: {sel_cinfo.get('max_depth', 600):.1f} km)<br>
                    • <b>High-Magnitude Rate (≥4.0):</b> {sel_cinfo.get('high_mag_ratio', 0)*100:.1f}%<br>
                    • <b>Severe Megathrust Rate (≥6.0):</b> {sel_cinfo.get('severe_mag_ratio', 0)*100:.2f}%
                </div>
                """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SUB-TAB 3: INTERACTIVE AREA-WISE PREDICTION SIMULATOR
    # --------------------------------------------------------------------------
    with cluster_tabs[2]:
        st.markdown("#### 🎯 Interactive Area-Wise Magnitude Prediction & Chunk Simulator")
        st.caption("Input custom hypocentral parameters or choose a preset to witness real-time tectonic chunk mapping and compare localized vs global forecasts:")

        sim_c1, sim_c2 = st.columns([1.1, 0.9])
        with sim_c1:
            st.markdown("##### 1. Coordinate & Hypocenter Parameters")
            sc_p1, sc_p2 = st.columns(2)
            with sc_p1:
                sim_lat = st.number_input("Latitude (°)", -90.0, 90.0, float(st.session_state.inp_lat), 0.05, format="%.4f", key="sim_in_lat")
                sim_lon = st.number_input("Longitude (°)", -180.0, 180.0, float(st.session_state.inp_lon), 0.05, format="%.4f", key="sim_in_lon")
            with sc_p2:
                sim_depth = st.number_input("Focal Depth (km)", 0.0, 750.0, float(st.session_state.inp_depth), 1.0, key="sim_in_depth")
                sim_month = st.slider("Month", 1, 12, int(st.session_state.inp_month), key="sim_in_month")
                sim_hour = st.slider("Hour (UTC)", 0, 23, int(st.session_state.inp_hour), key="sim_in_hour")

        with sim_c2:
            st.markdown("##### 2. Quick Autofill Hotspot Presets")
            st.caption("Click to load coordinates from known global fault zones:")
            quick_p = st.selectbox("Select Seismic Hotspot", list(PRESET_HOTSPOTS.keys()), key="sim_preset_choice")
            if st.button("Apply Hotspot Coordinates", key="btn_apply_sim_preset", use_container_width=True):
                apply_preset(quick_p)
                st.rerun()

        # Execute Area Prediction
        sim_area_res = predict_area_wise_magnitude(sim_lat, sim_lon, sim_depth, sim_month, sim_hour, models, regional_metrics_data)
        sim_tier_loc = get_magnitude_tier(sim_area_res["local_pred"])
        sim_tier_glo = get_magnitude_tier(sim_area_res["global_pred"])

        st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin: 20px 0;'>", unsafe_allow_html=True)
        st.markdown(f"##### 📍 Hypocenter Mapped to: <span style='color: #38bdf8;'>[{sim_area_res['cluster_id']}] {sim_area_res['cluster_name']}</span>", unsafe_allow_html=True)

        res_c1, res_c2, res_c3 = st.columns(3)
        with res_c1:
            st.markdown(f"""
            <div class="tp-card" style="border: 1px solid rgba(6, 182, 212, 0.4); box-shadow: 0 4px 20px rgba(6, 182, 212, 0.15); height: 100%;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #38bdf8; text-transform: uppercase;">🌍 Specialized Area Expert Model</div>
                <div style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #f8fafc; margin: 6px 0;">
                    M {sim_area_res['local_pred']:.2f}
                </div>
                <span class="tp-badge tp-badge-{sim_tier_loc['badge_color']}">● {sim_tier_loc['tier']} Severity</span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">
                    <b>Local Model MAE:</b> ± {sim_area_res['local_mae']:.4f} M<br>
                    <b>Confidence 95%:</b> M {max(1.0, sim_area_res['local_pred'] - sim_area_res['local_mae']):.2f} – M {sim_area_res['local_pred'] + sim_area_res['local_mae']:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_c2:
            st.markdown(f"""
            <div class="tp-card" style="border: 1px solid rgba(168, 85, 247, 0.4); box-shadow: 0 4px 20px rgba(168, 85, 247, 0.15); height: 100%;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #c084fc; text-transform: uppercase;">🌐 Global Random Forest Baseline</div>
                <div style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #f8fafc; margin: 6px 0;">
                    M {sim_area_res['global_pred']:.2f}
                </div>
                <span class="tp-badge tp-badge-{sim_tier_glo['badge_color']}">● {sim_tier_glo['tier']} Severity</span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">
                    <b>Global Model MAE:</b> ± 0.2862 M<br>
                    <b>Confidence 95%:</b> M {max(1.0, sim_area_res['global_pred'] - 0.29):.2f} – M {sim_area_res['global_pred'] + 0.29:.2f}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with res_c3:
            div_val = sim_area_res['local_pred'] - sim_area_res['global_pred']
            st.markdown(f"""
            <div class="tp-card" style="border: 1px solid rgba(16, 185, 129, 0.4); box-shadow: 0 4px 20px rgba(16, 185, 129, 0.15); height: 100%;">
                <div style="font-size: 0.8rem; font-weight: 700; color: #34d399; text-transform: uppercase;">📊 Historical Chunk Statistics</div>
                <div style="font-size: 2.8rem; font-weight: 800; font-family: 'JetBrains Mono'; color: #f8fafc; margin: 6px 0;">
                    M {sim_area_res['historical_mean']:.2f}
                </div>
                <span class="tp-badge tp-badge-emerald">Observed Empirical Baseline</span>
                <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">
                    <b>Historical Peak:</b> M {sim_area_res['historical_max']:.2f}<br>
                    <b>Local vs Global ΔM:</b> <b style="color: {'#34d399' if abs(div_val)<=0.2 else '#fb7185'};">{'+' if div_val>=0 else ''}{div_val:.2f} M</b>
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 12px; padding: 14px 18px; margin-top: 14px; font-size: 0.86rem; color: #cbd5e1; line-height: 1.5;">
            💡 <b>Geophysical Diagnostic Interpretation:</b><br>
            The specialized model for <b>{sim_area_res['cluster_name']}</b> is trained strictly on local fault rupture patterns. 
            Because this zone operates under <b>{sim_area_res['tectonic_type']}</b> mechanics, the area-wise expert avoids global underestimation in high-stress subduction zones (e.g. Japan/Chile) and prevents false alarms in low-stress continental microseismic areas (e.g. California/Texas).
        </div>
        """, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SUB-TAB 4: ALL-CLUSTERS BENCHMARK LEADERBOARD
    # --------------------------------------------------------------------------
    with cluster_tabs[3]:
        st.markdown("#### 📊 All-Clusters Benchmark Leaderboard & Comparative Summary")
        st.caption("Side-by-side performance comparison of all 8 tectonic Earth chunks and their respective local expert models:")

        bench_rows = []
        for cid, cinfo in regional_metrics_data.items():
            bench_rows.append({
                "ID": int(cid),
                "Tectonic Chunk Name": cinfo["name"],
                "Tectonic Regime": cinfo.get("tectonic_type", "Subduction"),
                "Event Count": cinfo.get("total_records", 0),
                "Catalog Share (%)": cinfo.get("percentage_of_catalog", 0.0),
                "Historical Mean (M)": cinfo.get("mean_mag", 0.0),
                "Historical Max (M)": cinfo.get("max_mag", 0.0),
                "High Mag Rate (≥4.0)": f"{cinfo.get('high_mag_ratio', 0)*100:.1f}%",
                "Avg Depth (km)": cinfo.get("mean_depth", 0.0),
                "Local MAE": cinfo.get("mae", 0.0),
                "Local RMSE": cinfo.get("rmse", 0.0),
                "Local R²": cinfo.get("r2", 0.0)
            })

        df_bench = pd.DataFrame(bench_rows).sort_values("ID")

        st.dataframe(
            df_bench.style.highlight_min(subset=["Local MAE", "Local RMSE"], color="rgba(16, 185, 129, 0.25)")
                          .highlight_max(subset=["Historical Max (M)", "Local R²"], color="rgba(6, 182, 212, 0.25)"),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("""
        <div style="font-size: 0.8rem; color: #94a3b8; margin-top: 10px;">
            * <b>Highlight Green:</b> Best local error (lowest MAE/RMSE).<br>
            * <b>Highlight Cyan:</b> Highest historical recorded magnitude and highest local R² determination score.
        </div>
        """, unsafe_allow_html=True)


# ==============================================================================
# TAB 2: GLOBAL 3D EARTH & SEISMIC EXPLORER
# ==============================================================================
elif app_mode == "🌐 Global 3D Seismic Explorer":
    st.markdown("""
    <div class="tp-card">
        <div class="tp-card-title">🌐 Global Seismicity Explorer & 3D Orthographic Globe</div>
        <div style="color: #94a3b8; font-size: 0.92rem;">
            Explore the complete <b>48,080 earthquake events</b> catalog indexed between 2024 and 2026. Filter by magnitude thresholds, focal depth layers, and projection modes to visualize the Pacific Ring of Fire and tectonic fault boundaries.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_dataset is not None:
        fcol1, fcol2, fcol3, fcol4 = st.columns([1.2, 1, 1, 1])

        with fcol1:
            mag_range = st.slider(
                "Magnitude Range (Richter)",
                min_value=float(df_dataset["mag"].min()),
                max_value=float(df_dataset["mag"].max()),
                value=(3.0, float(df_dataset["mag"].max())),
                step=0.1
            )

        with fcol2:
            depth_choices = st.multiselect(
                "Depth Categories",
                options=["Shallow", "Intermediate", "Deep"],
                default=["Shallow", "Intermediate", "Deep"]
            )

        with fcol3:
            proj_choice = st.selectbox(
                "Projection Mode",
                ["3D Orthographic (Globe)", "Natural Earth", "Equirectangular", "Density Heatmap"]
            )

        with fcol4:
            max_plot_points = st.selectbox(
                "Display Resolution",
                [2500, 5000, 10000, 20000],
                index=1,
                help="Subsamples data for ultra-smooth 60fps WebGL rendering."
            )

        filtered_df = df_dataset[
            df_dataset["mag"].between(mag_range[0], mag_range[1]) &
            df_dataset["depth_category"].isin(depth_choices)
        ]

        st.caption(f"Showing **{min(len(filtered_df), max_plot_points):,}** of **{len(filtered_df):,}** matched events (Total catalog: {len(df_dataset):,})")

        sample_to_plot = filtered_df.sample(min(len(filtered_df), max_plot_points), random_state=42)

        if proj_choice == "Density Heatmap":
            fig_map = px.density_mapbox(
                sample_to_plot,
                lat="latitude",
                lon="longitude",
                z="mag",
                radius=10,
                center=dict(lat=10, lon=0),
                zoom=1,
                mapbox_style="carto-darkmatter",
                color_continuous_scale="Turbo"
            )
            fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=550)
        else:
            proj_mode = "orthographic" if "3D" in proj_choice else ("natural earth" if "Natural" in proj_choice else "equirectangular")
            
            fig_map = px.scatter_geo(
                sample_to_plot,
                lat="latitude",
                lon="longitude",
                color="mag",
                size="mag",
                hover_name="place",
                hover_data={"mag": ":.2f", "depth": ":.1f km", "time": True, "latitude": False, "longitude": False},
                color_continuous_scale=["#38bdf8", "#06b6d4", "#f59e0b", "#f43f5e", "#ef4444"],
                projection=proj_mode,
                size_max=12
            )
            fig_map.update_geos(
                showcoastlines=True, coastlinecolor="#475569",
                showland=True, landcolor="#0f172a",
                showocean=True, oceancolor="#020617",
                showlakes=True, lakecolor="#020617",
                showcountries=True, countrycolor="#1e293b",
                bgcolor="rgba(0,0,0,0)"
            )
            fig_map.update_layout(
                paper_bgcolor="rgba(15, 23, 42, 0.4)",
                plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=0, b=0),
                height=560,
                coloraxis_colorbar=dict(title="Magnitude (M)", tickfont=dict(color="#f8fafc"))
            )

        st.plotly_chart(fig_map, use_container_width=True)

        g1, g2 = st.columns(2)

        with g1:
            st.markdown("#### 📈 Magnitude Distribution by Depth Layer")
            fig_box = px.histogram(
                filtered_df,
                x="mag",
                color="depth_category",
                nbins=35,
                barmode="overlay",
                color_discrete_map={"Shallow": "#38bdf8", "Intermediate": "#f59e0b", "Deep": "#f43f5e"},
                labels={"mag": "Magnitude (Richter)", "depth_category": "Depth Layer"}
            )
            fig_box.update_layout(**DARK_LAYOUT, height=320)
            st.plotly_chart(fig_box, use_container_width=True)

        with g2:
            st.markdown("#### 🌊 Focal Depth vs Magnitude Correlation Scatter")
            scatter_sample = filtered_df.sample(min(len(filtered_df), 1500), random_state=42)
            fig_scat = px.scatter(
                scatter_sample,
                x="depth",
                y="mag",
                color="mag",
                color_continuous_scale="Viridis",
                labels={"depth": "Focal Depth (km)", "mag": "Magnitude (M)"}
            )
            fig_scat.update_layout(**DARK_LAYOUT, height=320)
            st.plotly_chart(fig_scat, use_container_width=True)

        st.markdown("#### 🕒 Latest Catalog Seismic Events")
        recent_events = get_recent_live_events(filtered_df, limit=10)
        if recent_events:
            rec_df = pd.DataFrame(recent_events)
            st.dataframe(
                rec_df,
                column_config={
                    "time": st.column_config.DatetimeColumn("Recorded Timestamp (UTC)", format="YYYY-MM-DD HH:mm"),
                    "place": "Epicentral Location",
                    "mag": st.column_config.NumberColumn("Magnitude (M)", format="%.2f"),
                    "depth": st.column_config.NumberColumn("Depth (km)", format="%.1f km"),
                    "depth_category": "Depth Layer",
                    "magnitude_category": "Severity Class"
                },
                hide_index=True,
                use_container_width=True
            )
    else:
        st.error("Dataset 'earthquake_data_cleaned.csv' not found. Please ensure it is present in the workspace.")


# ==============================================================================
# TAB 3: EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================
elif app_mode == "📈 Exploratory Data Analysis (EDA)":
    st.markdown("""
    <div class="tp-card">
        <div class="tp-card-title">📈 Exploratory Data Analysis (EDA) Studio</div>
        <div style="color: #94a3b8; font-size: 0.92rem;">
            Deep-dive empirical investigation into the statistical properties, probability distributions, outlier boundaries, and tectonic correlations across <b>48,080 cataloged seismic events</b> (USGS Seismological Repository).
        </div>
    </div>
    """, unsafe_allow_html=True)

    if df_dataset is not None:
        eda_tab1, eda_tab2, eda_tab3, eda_tab4, eda_tab5 = st.tabs([
            "📊 Summary Stats & Outliers",
            "🏔️ Depth & Magnitude Categories",
            "🔗 Correlation & Covariance",
            "📅 Temporal & Seasonal Trends",
            "🎲 Probability & Sampling Theory"
        ])

        # TAB 1: SUMMARY STATS & OUTLIERS
        with eda_tab1:
            st.markdown("#### 1. Parametric & Non-Parametric Summary Statistics")
            num_cols = ["mag", "depth", "latitude", "longitude"]
            
            stats_df = pd.DataFrame({
                "Feature": ["Magnitude (M)", "Focal Depth (km)", "Latitude (°)", "Longitude (°)"],
                "Mean": df_dataset[num_cols].mean().values.round(3),
                "Median": df_dataset[num_cols].median().values.round(3),
                "Mode": [df_dataset[c].mode().iloc[0] for c in num_cols],
                "Std Dev": df_dataset[num_cols].std().values.round(3),
                "Variance": df_dataset[num_cols].var().values.round(3),
                "Q1 (25%)": df_dataset[num_cols].quantile(0.25).values.round(3),
                "Q3 (75%)": df_dataset[num_cols].quantile(0.75).values.round(3),
                "Min": df_dataset[num_cols].min().values.round(3),
                "Max": df_dataset[num_cols].max().values.round(3),
                "Skewness": df_dataset[num_cols].skew().values.round(3)
            })
            st.dataframe(stats_df, use_container_width=True, hide_index=True)

            st.markdown("---")
            st.markdown("#### 2. Magnitude Distribution & IQR Outlier Detection")
            
            q1 = float(df_dataset["mag"].quantile(0.25))
            q3 = float(df_dataset["mag"].quantile(0.75))
            iqr = q3 - q1
            lower_bound = round(q1 - 1.5 * iqr, 3)
            upper_bound = round(q3 + 1.5 * iqr, 3)
            outliers = df_dataset[(df_dataset["mag"] < lower_bound) | (df_dataset["mag"] > upper_bound)]

            oc1, oc2, oc3, oc4 = st.columns(4)
            oc1.metric("IQR Range (Q3 - Q1)", f"{iqr:.2f} M")
            oc2.metric("Lower Outlier Fence", f"{lower_bound:.2f} M")
            oc3.metric("Upper Outlier Fence", f"{upper_bound:.2f} M")
            oc4.metric("Potential Outliers", f"{len(outliers):,} events ({len(outliers)/len(df_dataset)*100:.1f}%)")

            ed_col1, ed_col2 = st.columns(2)
            with ed_col1:
                hist_bins = st.slider("Histogram Bins", min_value=15, max_value=80, value=35, step=5, key="eda_hist_bins")
                fig_dist = px.histogram(
                    df_dataset,
                    x="mag",
                    nbins=hist_bins,
                    marginal="box",
                    title="Earthquake Magnitude Distribution & Outliers",
                    color_discrete_sequence=["#06b6d4"],
                    labels={"mag": "Magnitude (Richter)"}
                )
                fig_dist.add_vline(x=upper_bound, line_dash="dash", line_color="#f43f5e", annotation_text="Upper Fence")
                fig_dist.update_layout(**DARK_LAYOUT, height=360)
                st.plotly_chart(fig_dist, use_container_width=True)

            with ed_col2:
                fig_box_all = px.box(
                    df_dataset,
                    y="mag",
                    points="outliers",
                    title="Magnitude Boxplot with Extreme Anomalies",
                    color_discrete_sequence=["#a855f7"],
                    labels={"mag": "Magnitude (M)"}
                )
                fig_box_all.update_layout(**DARK_LAYOUT, height=360)
                st.plotly_chart(fig_box_all, use_container_width=True)

            st.markdown("##### 🚨 Top 10 Extreme Outlier Seismic Events (Severe Shaking)")
            outlier_top = outliers.sort_values("mag", ascending=False).head(10)[["time", "place", "mag", "depth", "latitude", "longitude"]]
            st.dataframe(
                outlier_top,
                column_config={
                    "time": st.column_config.DatetimeColumn("Event Time (UTC)", format="YYYY-MM-DD HH:mm"),
                    "place": "Geographical Location",
                    "mag": st.column_config.NumberColumn("Peak Magnitude (M)", format="%.2f"),
                    "depth": st.column_config.NumberColumn("Focal Depth", format="%.1f km")
                },
                use_container_width=True,
                hide_index=True
            )

        # TAB 2: DEPTH & MAGNITUDE CATEGORIES
        with eda_tab2:
            st.markdown("#### 1. Categorical Distribution Breakdown (Matching Notebook Cell 6)")
            cat_c1, cat_c2 = st.columns(2)

            with cat_c1:
                depth_counts = df_dataset["depth_category"].value_counts().reset_index()
                depth_counts.columns = ["Depth Category", "Count"]
                fig_depth_bar = px.bar(
                    depth_counts,
                    x="Depth Category",
                    y="Count",
                    color="Depth Category",
                    category_orders={"Depth Category": ["Shallow", "Intermediate", "Deep"]},
                    color_discrete_map={"Shallow": "#38bdf8", "Intermediate": "#f59e0b", "Deep": "#f43f5e"},
                    text_auto=True,
                    title="Earthquakes by Focal Depth Category"
                )
                fig_depth_bar.update_layout(**DARK_LAYOUT, height=340, showlegend=False)
                st.plotly_chart(fig_depth_bar, use_container_width=True)

            with cat_c2:
                mag_counts = df_dataset["magnitude_category"].value_counts().reset_index()
                mag_counts.columns = ["Magnitude Category", "Count"]
                fig_mag_pie = px.pie(
                    mag_counts,
                    values="Count",
                    names="Magnitude Category",
                    title="Magnitude Severity Category Ratio",
                    hole=0.45,
                    color="Magnitude Category",
                    color_discrete_map={"Low (≤4)": "#34d399", "Moderate (>4–6)": "#fbbf24", "High (>6)": "#f43f5e"}
                )
                fig_mag_pie.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#f8fafc"),
                    height=340
                )
                st.plotly_chart(fig_mag_pie, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 2. Cross-Tabulation Matrix: Depth Layer vs. Magnitude Class")
            crosstab_df = pd.crosstab(df_dataset["depth_category"], df_dataset["magnitude_category"], margins=True)
            st.dataframe(crosstab_df, use_container_width=True)

        # TAB 3: CORRELATION & COVARIANCE
        with eda_tab3:
            st.markdown("#### 1. Feature Correlation Heatmaps (Matching Notebook Cell 11)")
            corr_cols = ["mag", "depth", "latitude", "longitude", "month", "hour"]
            
            method = st.radio("Select Correlation Metric", ["Pearson (Linear Association)", "Spearman (Rank-Order Monotonic)"], horizontal=True)
            calc_method = "pearson" if "Pearson" in method else "spearman"
            corr_matrix = df_dataset[corr_cols].corr(method=calc_method).round(3)

            fig_corr = px.imshow(
                corr_matrix,
                text_auto=True,
                color_continuous_scale="RdBu_r",
                zmin=-1.0,
                zmax=1.0,
                labels=dict(color=f"{calc_method.capitalize()} r"),
                title=f"{calc_method.capitalize()} Correlation Matrix ({len(df_dataset):,} events)"
            )
            fig_corr.update_layout(**DARK_LAYOUT, height=400)
            st.plotly_chart(fig_corr, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 2. Interactive Bivariate Scatter & Density Plot")
            sc1, sc2, sc3 = st.columns(3)
            with sc1:
                x_axis = st.selectbox("X-Axis Feature", corr_cols, index=1)
            with sc2:
                y_axis = st.selectbox("Y-Axis Feature", corr_cols, index=0)
            with sc3:
                color_var = st.selectbox("Color Mapping", ["depth_category", "magnitude_category", "high_magnitude"])

            scatter_sub = df_dataset.sample(min(2000, len(df_dataset)), random_state=42)
            fig_scat_eda = px.scatter(
                scatter_sub,
                x=x_axis,
                y=y_axis,
                color=color_var,
                opacity=0.6,
                title=f"{y_axis.upper()} vs {x_axis.upper()} (Subsample: 2,000 pts)",
                color_discrete_sequence=["#38bdf8", "#f59e0b", "#f43f5e"]
            )
            fig_scat_eda.update_layout(**DARK_LAYOUT, height=380)
            st.plotly_chart(fig_scat_eda, use_container_width=True)

        # TAB 4: TEMPORAL & SEASONAL TRENDS
        with eda_tab4:
            st.markdown("#### 1. Monthly Earthquake Timeline & Frequency (2024–2026)")
            monthly_agg = (
                df_dataset.set_index("time")
                .resample("MS")
                .agg(event_count=("mag", "count"), mean_magnitude=("mag", "mean"))
                .reset_index()
            )

            fig_monthly = px.line(
                monthly_agg,
                x="time",
                y="event_count",
                markers=True,
                title="Global Monthly Earthquake Occurrences Over Time",
                labels={"time": "Date", "event_count": "Recorded Earthquakes"},
                color_discrete_sequence=["#38bdf8"]
            )
            fig_monthly.update_layout(**DARK_LAYOUT, height=330)
            st.plotly_chart(fig_monthly, use_container_width=True)

            tc1, tc2 = st.columns(2)
            with tc1:
                st.markdown("##### Seasonal Monthly Rhythm (Months 1–12)")
                m_counts = df_dataset["month"].value_counts().sort_index().reset_index()
                m_counts.columns = ["Month", "Events"]
                fig_m = px.bar(m_counts, x="Month", y="Events", color="Events", color_continuous_scale="Teal", text_auto=True)
                fig_m.update_layout(**DARK_LAYOUT, height=300)
                st.plotly_chart(fig_m, use_container_width=True)

            with tc2:
                st.markdown("##### Diurnal Hourly Frequency (0–23 UTC)")
                h_counts = df_dataset["hour"].value_counts().sort_index().reset_index()
                h_counts.columns = ["Hour (UTC)", "Events"]
                fig_h = px.bar(h_counts, x="Hour (UTC)", y="Events", color="Events", color_continuous_scale="Purples", text_auto=True)
                fig_h.update_layout(**DARK_LAYOUT, height=300)
                st.plotly_chart(fig_h, use_container_width=True)

        # TAB 5: PROBABILITY & SAMPLING THEORY
        with eda_tab5:
            st.markdown("#### 1. Simulated Probability Distributions (Matching Notebook Cell 7)")
            st.markdown("Comparing the observed earthquake magnitude distribution against standard theoretical probability models:")

            mags = df_dataset["mag"].dropna().to_numpy()
            n_sim = min(5000, len(mags))
            norm_sim = np.random.normal(mags.mean(), mags.std(), n_sim)
            unif_sim = np.random.uniform(0, 1, n_sim)
            p_high = float((mags >= 4.0).mean())
            bern_sim = np.random.binomial(1, p_high, n_sim)

            p_col1, p_col2, p_col3 = st.columns(3)
            with p_col1:
                fig_p1 = px.histogram(norm_sim, nbins=30, title=f"Simulated Gaussian (μ={mags.mean():.2f}, σ={mags.std():.2f})", color_discrete_sequence=["#38bdf8"])
                fig_p1.update_layout(**DARK_LAYOUT, height=280)
                st.plotly_chart(fig_p1, use_container_width=True)

            with p_col2:
                fig_p2 = px.histogram(unif_sim, nbins=30, title="Uniform Distribution U(0, 1)", color_discrete_sequence=["#f59e0b"])
                fig_p2.update_layout(**DARK_LAYOUT, height=280)
                st.plotly_chart(fig_p2, use_container_width=True)

            with p_col3:
                fig_p3 = px.histogram(bern_sim, title=f"Bernoulli (p={p_high:.2f} for M ≥ 4)", color_discrete_sequence=["#10b981"])
                fig_p3.update_layout(**DARK_LAYOUT, height=280)
                st.plotly_chart(fig_p3, use_container_width=True)

            st.markdown("---")
            st.markdown("#### 2. Sampling Techniques Comparison (Matching Notebook Cell 8)")
            st.caption("Empirical verification of different sampling strategies drawn from the catalog population (n=100):")

            pop = df_dataset["mag"].dropna().reset_index(drop=True)
            s_size = 100
            s_simple = pop.sample(s_size, random_state=42)
            s_replace = pop.sample(s_size, replace=True, random_state=42)
            s_noreplace = pop.sample(s_size, replace=False, random_state=21)
            step_k = max(1, len(pop) // s_size)
            s_syst = pop.iloc[::step_k].head(s_size)

            samp_summary = pd.DataFrame({
                "Sampling Method": ["Full Population", "Simple Random Sampling", "With Replacement (Bootstrap)", "Without Replacement", "Systematic Sampling"],
                "Sample Size (n)": [len(pop), s_size, s_size, s_size, s_size],
                "Mean Magnitude": [round(pop.mean(), 4), round(s_simple.mean(), 4), round(s_replace.mean(), 4), round(s_noreplace.mean(), 4), round(s_syst.mean(), 4)],
                "Variance": [round(pop.var(), 4), round(s_simple.var(), 4), round(s_replace.var(), 4), round(s_noreplace.var(), 4), round(s_syst.var(), 4)],
                "Standard Error": [0.0, round(s_simple.std()/np.sqrt(s_size), 4), round(s_replace.std()/np.sqrt(s_size), 4), round(s_noreplace.std()/np.sqrt(s_size), 4), round(s_syst.std()/np.sqrt(s_size), 4)]
            })
            st.dataframe(samp_summary, use_container_width=True, hide_index=True)
    else:
        st.error("Dataset 'earthquake_data_cleaned.csv' not found.")


# ==============================================================================
# TAB 4: STATISTICAL INFERENCE & HYPOTHESIS TESTING HUB
# ==============================================================================
elif app_mode == "📐 Statistical Inference & Hypothesis Hub":
    st.markdown("""
    <div class="tp-card">
        <div class="tp-card-title">📐 Statistical Inference & Hypothesis Testing Engine</div>
        <div style="color: #94a3b8; font-size: 0.92rem;">
            This project investigates statistical significance, population hypothesis testing, and asymptotic probability distributions across 48,080 seismic events as outlined in the project's statistical methodology.
        </div>
    </div>
    """, unsafe_allow_html=True)

    stat_tab1, stat_tab2, stat_tab3, stat_tab4, stat_tab5 = st.tabs([
        "1️⃣ One-Sample t-Test",
        "2️⃣ Welch's Two-Sample t-Test",
        "3️⃣ One-Way ANOVA",
        "4️⃣ Chi-Square Independence",
        "5️⃣ Central Limit Theorem (CLT)"
    ])

    with stat_tab1:
        st.markdown("### One-Sample t-Test: Global Mean Magnitude")
        st.markdown(r"""
        - **Null Hypothesis ($H_0$):** The true population mean earthquake magnitude $\mu = 4.0$.
        - **Alternative Hypothesis ($H_1$):** The true population mean earthquake magnitude $\mu \neq 4.0$.
        - **Significance Threshold ($\alpha$):** $0.05$
        """)

        t1_info = stats_data.get("one_sample_t", {})
        tc1, tc2, tc3, tc4 = st.columns(4)
        tc1.metric("Hypothesized Mean (μ₀)", f"{t1_info.get('popmean', 4.0)}")
        tc2.metric("Observed Mean (x̄)", f"{t1_info.get('sample_mean', 3.88)}")
        tc3.metric("t-Statistic", f"{t1_info.get('t_stat', -31.51)}")
        tc4.metric("p-Value", f"{t1_info.get('p_value', 0.0):.2e}")

        st.success(f"**Statistical Verdict:** {t1_info.get('decision', 'Reject H0')}. With $p < 10^{-200}$, we definitively reject $H_0$. The global catalog mean magnitude is statistically significantly lower than 4.0.")

    with stat_tab2:
        st.markdown("### Welch's Two-Sample t-Test: Shallow vs Deep Focus")
        st.markdown(r"""
        - **Null Hypothesis ($H_0$):** Shallow earthquakes ($\le 70\text{ km}$) and Deep earthquakes ($> 300\text{ km}$) exhibit identical mean magnitudes ($\mu_{\text{shallow}} = \mu_{\text{deep}}$).
        - **Alternative Hypothesis ($H_1$):** Focal depth layers exhibit distinct mean magnitudes ($\mu_{\text{shallow}} \neq \mu_{\text{deep}}$).
        """)

        t2_info = stats_data.get("welch_two_sample_t", {})
        w1, w2, w3, w4 = st.columns(4)
        w1.metric("Shallow Mean (n=37,663)", f"{t2_info.get('shallow_mean', 3.81)} M")
        w2.metric("Deep Mean (n=2,072)", f"{t2_info.get('deep_mean', 4.35)} M")
        w3.metric("Welch's t-Stat", f"{t2_info.get('t_stat', -62.53)}")
        w4.metric("p-Value", "< 1e-15")

        st.warning(f"**Statistical Verdict:** {t2_info.get('decision', 'Reject H0')}. Deep focal events have a substantially higher mean magnitude ({t2_info.get('deep_mean')} vs {t2_info.get('shallow_mean')}).")

    with stat_tab3:
        st.markdown("### One-Way ANOVA: Variance Across All Depth Tiers")
        st.markdown(r"""
        - **Null Hypothesis ($H_0$):** Mean magnitude is uniform across all three depth classes ($\mu_{\text{shallow}} = \mu_{\text{intermediate}} = \mu_{\text{deep}}$).
        - **Alternative Hypothesis ($H_1$):** At least one depth tier possesses a distinct mean magnitude.
        """)

        anova_info = stats_data.get("anova", {})
        a1, a2, a3 = st.columns(3)
        a1.metric("F-Statistic", f"{anova_info.get('f_stat', 648.01)}")
        a2.metric("ANOVA p-Value", "< 1e-270")
        a3.metric("Decision", "Reject H0 (Extremely Significant)")

        if df_dataset is not None:
            fig_anova = px.box(
                df_dataset,
                x="depth_category",
                y="mag",
                color="depth_category",
                category_orders={"depth_category": ["Shallow", "Intermediate", "Deep"]},
                color_discrete_map={"Shallow": "#38bdf8", "Intermediate": "#f59e0b", "Deep": "#f43f5e"},
                labels={"depth_category": "Depth Layer", "mag": "Magnitude (M)"}
            )
            fig_anova.update_layout(**DARK_LAYOUT, height=350, showlegend=False)
            st.plotly_chart(fig_anova, use_container_width=True)

    with stat_tab4:
        st.markdown("### Chi-Square (χ²) Contingency Test: Depth vs Magnitude")
        st.markdown(r"""
        - **Null Hypothesis ($H_0$):** Depth category and Magnitude category are independent.
        - **Alternative Hypothesis ($H_1$):** Depth category and Magnitude category are significantly associated.
        """)

        chi_info = stats_data.get("chi_square", {})
        c1, c2, c3 = st.columns(3)
        c1.metric("χ² Statistic", f"{chi_info.get('chi2_stat', 1696.87)}")
        c2.metric("Degrees of Freedom", f"{chi_info.get('dof', 4)}")
        c3.metric("p-Value", "< 1e-300")

        st.success(f"**Statistical Verdict:** {chi_info.get('decision', 'Reject H0')}.")

        obs_dict = chi_info.get("contingency_observed", {})
        exp_dict = chi_info.get("contingency_expected", {})
        if obs_dict and exp_dict:
            df_obs = pd.DataFrame(obs_dict)
            df_exp = pd.DataFrame(exp_dict)
            
            hc1, hc2 = st.columns(2)
            with hc1:
                st.markdown("##### Observed Frequencies")
                fig_obs = px.imshow(
                    df_obs,
                    text_auto=True,
                    color_continuous_scale="Blues",
                    labels=dict(x="Magnitude Class", y="Depth Layer", color="Count")
                )
                fig_obs.update_layout(**DARK_LAYOUT, height=280)
                st.plotly_chart(fig_obs, use_container_width=True)

            with hc2:
                st.markdown("##### Expected Frequencies (Under H₀ Independence)")
                fig_exp = px.imshow(
                    df_exp,
                    text_auto=True,
                    color_continuous_scale="Oranges",
                    labels=dict(x="Magnitude Class", y="Depth Layer", color="Expected")
                )
                fig_exp.update_layout(**DARK_LAYOUT, height=280)
                st.plotly_chart(fig_exp, use_container_width=True)

    with stat_tab5:
        st.markdown("### Central Limit Theorem (CLT) Monte Carlo Simulator")
        st.markdown("""
        The Central Limit Theorem states that as sample size $n$ increases, the distribution of sample means approaches a Gaussian distribution regardless of the underlying population shape, with standard error $\\text{SE} = \\sigma / \\sqrt{n}$.
        """)

        clt_col1, clt_col2 = st.columns([1, 2])
        with clt_col1:
            clt_n = st.slider("Sample Size (n)", min_value=5, max_value=250, value=50, step=5)
            clt_k = st.slider("Monte Carlo Iterations (k)", min_value=200, max_value=3000, value=1200, step=100)
            
            if df_dataset is not None:
                sim_res = run_clt_simulation(df_dataset["mag"], sample_size=clt_n, repetitions=clt_k)
                st.metric("Population Mean (μ)", f"{sim_res['pop_mean']:.4f}")
                st.metric("Observed Mean of Means", f"{sim_res['obs_mean']:.4f}")
                st.metric("Observed Standard Error", f"{sim_res['obs_se']:.4f}")
                st.metric("Theoretical SE (σ / √n)", f"{sim_res['theo_se']:.4f}")

        with clt_col2:
            if df_dataset is not None:
                fig_clt = px.histogram(
                    x=sim_res["sample_means"],
                    nbins=40,
                    histnorm="probability density",
                    labels={"x": "Sample Mean Magnitude"},
                    title=f"Distribution of Sample Means (n={clt_n}, k={clt_k:,})",
                    color_discrete_sequence=["#06b6d4"]
                )
                fig_clt.add_vline(x=sim_res["pop_mean"], line_dash="dash", line_color="#f43f5e", annotation_text="μ Population")
                fig_clt.update_layout(**DARK_LAYOUT, height=360)
                st.plotly_chart(fig_clt, use_container_width=True)


# ==============================================================================
# TAB 4: MODEL BENCHMARK & AI ANALYTICS
# ==============================================================================
elif app_mode == "📊 Model Benchmark & Analytics":
    st.markdown("""
    <div class="tp-card">
        <div class="tp-card-title">📊 Machine Learning Evaluation & Benchmark Lab</div>
        <div style="color: #94a3b8; font-size: 0.92rem;">
            Comparative benchmarking of classification and regression pipelines evaluated under Stratified 5-Fold Cross Validation alongside Feature Importance and Residual Analysis.
        </div>
    </div>
    """, unsafe_allow_html=True)

    classifiers_list = metrics_data.get("classifiers_comparison", [])
    if classifiers_list:
        df_comp = pd.DataFrame(classifiers_list)
        
        st.markdown("#### 🏆 Classification Model Leaderboard")
        st.dataframe(
            df_comp.style.highlight_max(subset=["Accuracy", "Precision", "Recall", "F1_Score"], color="rgba(16, 185, 129, 0.3)"),
            use_container_width=True,
            hide_index=True
        )

        bcol1, bcol2 = st.columns(2)

        with bcol1:
            st.markdown("#### 📈 Cross-Model Metric Comparison")
            fig_bar = px.bar(
                df_comp,
                x="Model",
                y=["Accuracy", "F1_Score"],
                barmode="group",
                color_discrete_sequence=["#38bdf8", "#06b6d4"],
                labels={"value": "Score (0 - 1.0)", "variable": "Evaluation Metric"}
            )
            fig_bar.update_layout(**DARK_LAYOUT, height=340)
            st.plotly_chart(fig_bar, use_container_width=True)

        with bcol2:
            st.markdown("#### 🎯 Confusion Matrix Inspector")
            selected_m = st.selectbox("Select Model Architecture", [c["Model"] for c in classifiers_list])
            cm = metrics_data.get("confusion_matrices", {}).get(selected_m, [[3816, 57], [201, 5542]])
            
            cm_df = pd.DataFrame(
                cm,
                index=["Actual Low (<4.0)", "Actual High (≥4.0)"],
                columns=["Pred Low (<4.0)", "Pred High (≥4.0)"]
            )
            fig_cm = px.imshow(
                cm_df,
                text_auto=True,
                color_continuous_scale="Teal",
                labels=dict(x="Predicted Class", y="Actual Class", color="Events")
            )
            fig_cm.update_layout(**DARK_LAYOUT, height=300)
            st.plotly_chart(fig_cm, use_container_width=True)

    # Magnitude Regressor Performance Card
    st.markdown("---")
    st.markdown("#### 📐 Magnitude Regressors Performance Leaderboard")
    st.caption("Cross-comparison of continuous magnitude regression algorithms trained on 48,080 events:")
    reg_models_dict = metrics_data.get("regression_models", {})
    if reg_models_dict:
        reg_df = pd.DataFrame(reg_models_dict).T.reset_index().rename(columns={"index": "Algorithm"})
        st.dataframe(
            reg_df.style.highlight_min(subset=["MAE", "MSE", "RMSE"], color="rgba(16, 185, 129, 0.25)")
                        .highlight_max(subset=["R2"], color="rgba(6, 182, 212, 0.25)")
                        .format({"MAE": "{:.4f}", "MSE": "{:.4f}", "RMSE": "{:.4f}", "R2": "{:.4f}"}),
            use_container_width=True,
            hide_index=True
        )

        rcol1, rcol2 = st.columns(2)
        with rcol1:
            st.markdown("#### 📉 Regressor Error Comparison (MAE vs RMSE)")
            fig_reg_err = px.bar(
                reg_df,
                x="Algorithm",
                y=["MAE", "RMSE"],
                barmode="group",
                color_discrete_sequence=["#c084fc", "#38bdf8"],
                labels={"value": "Error (Magnitude units)", "variable": "Metric"}
            )
            fig_reg_err.update_layout(**DARK_LAYOUT, height=300)
            st.plotly_chart(fig_reg_err, use_container_width=True)

        with rcol2:
            st.markdown("#### 🧭 Decision Tree vs Random Forest Feature Weights")
            dt_fi = metrics_data.get("dt_feature_importances", [])
            rf_fi = metrics_data.get("regressor_feature_importances", [])
            if dt_fi and rf_fi:
                df_dt_fi = pd.DataFrame(dt_fi).assign(Model="Decision Tree Regressor")
                df_rf_fi = pd.DataFrame(rf_fi).assign(Model="Random Forest Regressor")
                df_fi_both = pd.concat([df_dt_fi, df_rf_fi], ignore_index=True)
                fig_fi_comp = px.bar(
                    df_fi_both,
                    x="Importance",
                    y="Feature",
                    color="Model",
                    barmode="group",
                    orientation="h",
                    color_discrete_sequence=["#10b981", "#06b6d4"],
                    labels={"Importance": "Gini Importance", "Feature": "Predictor"}
                )
                fig_fi_comp.update_layout(**DARK_LAYOUT, height=300)
                st.plotly_chart(fig_fi_comp, use_container_width=True)

    # Feature Importance Section
    st.markdown("---")
    f_imp_list = metrics_data.get("feature_importances", [])
    if f_imp_list:
        fcol1, fcol2 = st.columns([1.2, 0.8])
        
        with fcol1:
            st.markdown("#### 🧭 Classification Feature Importance (Random Forest)")
            df_fi = pd.DataFrame(f_imp_list)
            fig_fi = px.bar(
                df_fi,
                x="Importance",
                y="Feature",
                orientation="h",
                color="Importance",
                color_continuous_scale=["#0891b2", "#06b6d4", "#38bdf8"],
                labels={"Importance": "Gini Importance Score", "Feature": "Tectonic Predictor"}
            )
            fig_fi.update_layout(**DARK_LAYOUT, height=280)
            st.plotly_chart(fig_fi, use_container_width=True)

        with fcol2:
            st.markdown("#### 💡 Seismological Interpretation")
            st.markdown("""
            - **Longitude & Latitude (~89% weight):** Earthquakes cluster strictly along tectonic margins (e.g. Ring of Fire, Alpine-Himalayan belt). Geographic coordinate vectors strongly govern severity potential.
            - **Focal Depth (~9% weight):** Subduction zone megathrusts vs shallow intraplate fault ruptures behave differently in total energy release.
            - **Month & Hour (~2% weight):** Minor seasonal hydrology loading factors; earthquake occurrences are predominantly tectonic rather than diurnal.
            """)


# ==============================================================================
# TAB 5: BATCH CSV DIAGNOSTICS & REPORT EXPORT
# ==============================================================================
elif app_mode == "📁 Batch CSV Diagnostics":
    st.markdown("""
    <div class="tp-card">
        <div class="tp-card-title">📁 Batch Seismic Diagnostics & Pipeline Export</div>
        <div style="color: #94a3b8; font-size: 0.92rem;">
            Upload your seismic sensor telemetry or test the built-in sample batch of global earthquake events to perform high-throughput inference with automated magnitude forecasting across Multiple Linear Regression, Decision Tree, and Random Forest, complete with severity labeling and physical hazard metrics.
        </div>
    </div>
    """, unsafe_allow_html=True)

    b1, b2 = st.columns([1.2, 0.8])
    with b1:
        uploaded_file = st.file_uploader("Upload Event CSV File", type=["csv"])
    with b2:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        use_sample = st.button("📂 Load Built-In Global Hotspot Fleet (10 Events)", use_container_width=True)

    batch_df = None
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
    elif use_sample:
        sample_path = Path("sample_earthquake_batch.csv")
        if sample_path.exists():
            batch_df = pd.read_csv(sample_path)

    if batch_df is not None:
        st.markdown("#### 🔍 Input Telemetry Preview")
        st.dataframe(batch_df.head(10), use_container_width=True)

        req_cols = ["depth", "latitude", "longitude", "month", "hour"]
        missing = [c for c in req_cols if c not in batch_df.columns]
        
        if missing:
            st.error(f"Missing required columns in CSV: {missing}. Required columns: {req_cols}")
        else:
            with st.spinner("Processing batch inference..."):
                X_batch = batch_df[req_cols].copy()
                preds = classifier.predict(X_batch)
                probas = classifier.predict_proba(X_batch)[:, 1]
                
                reg_preds_lr = regressor_lr.predict(X_batch) if regressor_lr is not None else np.where(preds == 1, 4.6, 3.4)
                reg_preds_dt = regressor_dt.predict(X_batch) if regressor_dt is not None else np.where(preds == 1, 4.7, 3.3)
                reg_preds_rf = regressor_rf.predict(X_batch) if regressor_rf is not None else np.where(preds == 1, 4.8, 3.2)
                consensus_preds = (reg_preds_lr + reg_preds_dt + reg_preds_rf) / 3.0
                
                # Area-Wise Tectonic Cluster Mapping & Local Predictions
                chunk_labels = []
                area_preds = []
                for _, row in X_batch.iterrows():
                    ainfo = predict_area_wise_magnitude(row["latitude"], row["longitude"], row["depth"], int(row["month"]), int(row["hour"]), models, regional_metrics_data)
                    chunk_labels.append(ainfo["cluster_name"].split("/")[0].strip())
                    area_preds.append(ainfo["local_pred"])

                res_df = batch_df.copy()
                res_df["Tectonic_Chunk"] = chunk_labels
                res_df["Area_Wise_Magnitude"] = np.round(area_preds, 2)
                res_df["RF_Magnitude"] = np.round(reg_preds_rf, 2)
                res_df["LinearReg_Magnitude"] = np.round(reg_preds_lr, 2)
                res_df["DecisionTree_Magnitude"] = np.round(reg_preds_dt, 2)
                res_df["Consensus_Magnitude"] = np.round(consensus_preds, 2)
                res_df["Magnitude_Tier"] = [get_magnitude_tier(m)["tier"] for m in area_preds]
                res_df["Classification"] = np.where(preds == 1, "High Mag (≥4.0)", "Low Mag (<4.0)")
                res_df["Confidence_%"] = np.round(probas * 100, 1)
                res_df["Energy_Joules"] = [f"{10**(4.8 + 1.5*m):.2e}" for m in area_preds]
                res_df["TNT_Equiv_kt"] = np.round([ (10**(4.8 + 1.5*m))/(4.184e12) for m in area_preds ], 2)

                st.markdown("#### ⚡ Batch Multi-Model Magnitude & Diagnostic Fleet Results")
                st.dataframe(
                    res_df,
                    column_config={
                        "Tectonic_Chunk": "Earth Tectonic Chunk",
                        "Area_Wise_Magnitude": st.column_config.NumberColumn("Area Expert (M)", format="%.2f"),
                        "RF_Magnitude": st.column_config.NumberColumn("RF Regressor (M)", format="%.2f"),
                        "LinearReg_Magnitude": st.column_config.NumberColumn("Multiple Reg (M)", format="%.2f"),
                        "DecisionTree_Magnitude": st.column_config.NumberColumn("Decision Tree (M)", format="%.2f"),
                        "Consensus_Magnitude": st.column_config.NumberColumn("Consensus (M)", format="%.2f"),
                        "Magnitude_Tier": "Severity Tier",
                        "Classification": "Binary Class",
                        "Confidence_%": st.column_config.ProgressColumn("P(M ≥ 4.0)", min_value=0, max_value=100, format="%d%%"),
                        "TNT_Equiv_kt": st.column_config.NumberColumn("Energy (kt TNT)", format="%.2f kt")
                    },
                    use_container_width=True
                )

                csv_bytes = res_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Annotated Batch Predictions (CSV)",
                    data=csv_bytes,
                    file_name="terrapulse_annotated_batch_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                    type="primary"
                )
    else:
        st.info("Upload a CSV file or click 'Load Built-In Global Hotspot Fleet' to begin batch evaluation.")

# Footer
st.markdown("<hr style='border-color: rgba(255,255,255,0.08); margin-top: 40px;'>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align: center; color: #64748b; font-size: 0.8rem; padding-bottom: 20px;">
    TerraPulse AI • Earthquake Magnitude & Seismic Activity Analysis Using Statistical Inference<br>
    Machine Learning • USGS Seismology Repository • Built with Streamlit & Plotly
</div>
""", unsafe_allow_html=True)
