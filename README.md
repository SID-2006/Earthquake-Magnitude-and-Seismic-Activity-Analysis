# 🌋 TerraPulse AI — Earthquake Seismic Intelligence & Magnitude Forecasting Suite

[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit%201.66-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Python](https://img.shields.io/badge/Python-3.14-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.9.0-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![USGS Data](https://img.shields.io/badge/USGS-48%2C080%20Events-006699?style=for-the-badge&logo=usgs&logoColor=white)](https://earthquake.usgs.gov)

An enterprise-grade, modern **Seismological Intelligence & Earthquake Magnitude Analytics Platform** built from the research in [`Eartquake_SI_project.ipynb`](file:///f:/Eartquake/Eartquake_SI_project.ipynb). TerraPulse AI transforms statistical inference models, machine learning pipelines, and USGS historical records into an interactive, real-time diagnostic web application.

---

## ✨ Key Modules & Capabilities

### 1. 🎯 Earthquake Magnitude Predictor & Diagnostics
- **Dedicated Magnitude Forecasting Engine**:
  - **Exact Richter / Moment Magnitude ($M$) Estimation**: High-precision continuous regression predicting exact event magnitude.
  - **Algorithm Selector**: Switch dynamically between:
    - **⚡ Multi-Model Comparison (Multiple Linear Regression vs Decision Tree vs Random Forest)**
    - **🌍 Area-Wise Regional Expert (Specialized Earth Chunk Model)**
    - **🌲 Random Forest Regressor (Ensemble Non-Linear Bagging)**
    - **🌳 Decision Tree Regressor (Recursive CART Tree Splitter)**
    - **📐 Multiple Linear Regression (Statistical Baseline / OLS Hyperplane)**
    - **🤝 Triple Consensus Ensemble (Weighted Model Blend)**
  - **Continuous Richter Scale Visual Position Indicator**: Plotly spectrum mapping the predicted magnitude across Micro ($<3.0$), Minor ($3.0-3.9$), Light ($4.0-4.9$), Moderate ($5.0-5.9$), Strong ($6.0-6.9$), Major ($7.0-7.9$), and Great ($\ge 8.0$).
  - **Confidence Interval**: 95% uncertainty interval based on held-out test residuals ($M \pm 0.28$).
- **Focal Depth Sensitivity & Attenuation Curve**:
  - Interactive multi-trace curve plotting predicted magnitude across focal depth (0 km to 600 km) for **Multiple Linear Regression**, **Decision Tree Regressor**, **Random Forest Regressor**, and the **Area-Wise Local Expert**.
- **Physical Hazard & Geophysical Impact Profile**:
  - **Peak Ground Acceleration (PGA)**: Estimated ground shaking percentage ($\%g$).
  - **Felt Area Radius**: Perceptible geographical radius in kilometers ($R \approx 10^{0.43M - 0.25}$).
  - **Radiated Seismic Energy (Gutenberg-Richter)**: Total energy release $\log_{10}(E) = 4.8 + 1.5 M$ in Joules and Kilotons TNT equivalent.
  - **Modified Mercalli Intensity (MMI)**: Epicentral ground-shaking scale (I to XII) with structural damage criteria.
  - **Tsunami Hazard Watch**: Submarine megathrust vs inland focal depth wave generation assessment.
- **Quick Tectonic Presets**: One-click autofill for global fault lines:
  - *Tokyo / Kanto (Japan)*, *San Andreas Fault (California)*, *Sumatra Megathrust (Indonesia)*, *Valdivia (Chile)*, *Himalayan Thrust (Nepal)*, *Reykjanes Rift (Iceland)*, *Tonga Deep Trench*.

### 2. 🌍 Tectonic Clusters & Earth Spatial Chunking Engine (Area-Wise Predictions)
- **Unit-Sphere 3D Spatial Clustering ($K=8$ Chunks)**:
  - Converts $(\text{lat}, \text{lon})$ to 3D Cartesian coordinates $(x, y, z)$ on the unit sphere to avoid meridian/pole distortions at $\pm 180^\circ$.
  - Partitions Earth into 8 geophysically coherent tectonic provinces:
    1. **NW Pacific / Kuril-Kamchatka-Japan Arc**: High-velocity subduction zone (Mean $M=4.50$, Max $8.80$, $99.7\%$ high mag).
    2. **Caribbean Basin & Antilles Fault System**: Complex transform / subduction margin (Mean $M=3.74$, Max $7.60$, $42.2\%$ high mag).
    3. **SE Asia / Sunda-Banda & Coral Triangle**: Sunda megathrust and Indonesian arc (Mean $M=4.50$, Max $7.70$, $98.6\%$ high mag).
    4. **Alpine-Himalayan & Mediterranean Belt**: Continental collision zone (Mean $M=4.39$, Max $7.10$, $97.8\%$ high mag, Lowest MAE: $\pm 0.2255$).
    5. **North America Continental & San Andreas**: Shallow strike-slip faulting and Basin & Range (Mean $M=3.02$, Max $7.00$, $8.6\%$ high mag, shallow $8.8\text{ km}$).
    6. **SW Pacific / Tonga-Kermadec-Fiji Trench**: Deepest Wadati-Benioff mantle subduction (Mean $M=4.52$, Max $7.30$, $99.4\%$ high mag, deep $204.9\text{ km}$).
    7. **South America / Andean Subduction & Chile**: Nazca subduction interface (Mean $M=4.53$, Max $7.60$, $98.2\%$ high mag).
    8. **North Pacific / Alaska-Aleutians & Hawaii**: Aleutian megathrust and volcanic swarms (Mean $M=3.01$, Max $7.30$, $6.3\%$ high mag).
- **Dedicated Regional Expert Models (`regional_cluster_models.joblib`)**:
  - Fits an isolated Random Forest Regressor tailored exclusively to each region's specific fault mechanics, rock elasticity, and depth-magnitude attenuation.
  - Slashes localized prediction errors to as low as **$\pm 0.225$ magnitude units**.
- **Interactive Tectonic Explorer & Simulator**:
  - 3D rotating globe / 2D projections color-coded by the 8 tectonic chunks with glowing centroid pins.
  - Deep cluster inspector with localized feature importance rankings (showing how focal depth vs coordinate strike behaves per chunk).
  - Real-time Area-Wise Prediction simulator showing local expert vs global baseline forecasts.

### 3. 🌐 Global 3D Seismic Explorer & Orthographic Earth
- **3D Rotating Orthographic Globe & 2D Projection**: Visualize all 48,080 recorded events across the Pacific Ring of Fire and mid-ocean ridges.
- **Multi-Layer Filtering**: Filter by Richter magnitude range, focal depth tiers (*Shallow* $\le 70\text{km}$, *Intermediate* $70-300\text{km}$, *Deep* $> 300\text{km}$), and map resolution.
- **Catalog Events Feed**: Instant inspectable table of recent seismic events with focal coordinates and timestamps.

### 3. 📈 Exploratory Data Analysis (EDA) Studio
Direct interactive implementation of the exploratory findings from [`Eartquake_SI_project.ipynb`](file:///f:/Eartquake/Eartquake_SI_project.ipynb):
- **Summary Statistics & Outlier Detection**: Parametric and non-parametric metrics (Mean, Median, Mode, Variance, Std Dev, Q1, Q3, Min, Max, Skewness) plus $1.5 \times \text{IQR}$ outlier detection table.
- **Depth & Magnitude Breakdown**: Focal depth distribution bars, magnitude severity donut chart, and 2D cross-tabulation matrix.
- **Feature Correlation & Covariance Matrix**: Annotated Pearson and Spearman heatmaps with interactive bivariate scatter inspector.
- **Temporal & Seasonal Seismicity Trends**: 2024–2026 monthly timeline, seasonal monthly rhythm (1–12), and diurnal hourly frequency (0–23 UTC).
- **Probability Distributions & Sampling Theory**: Empirical vs. simulated Gaussian, Uniform, and Bernoulli distributions alongside population vs. sampling technique comparisons.

### 4. 📐 Statistical Inference & Hypothesis Testing Hub
Realizes all hypothesis tests from the project's statistical methodology:
- **One-Sample t-Test**: Tests whether the global mean magnitude $\mu = 4.0$ ($t = -31.51, p < 10^{-200} \implies \text{Reject } H_0$).
- **Welch's Two-Sample t-Test**: Tests whether shallow vs deep focus events differ significantly in mean magnitude ($t = -62.53, p < 10^{-15} \implies \text{Reject } H_0$).
- **One-Way ANOVA (F-Test)**: Compares magnitude variance across all three depth layers ($F = 648.01, p < 10^{-270}$).
- **Chi-Square ($\chi^2$) Contingency Test**: Depth layer vs Magnitude class independence test ($\chi^2 = 1,696.87, p < 10^{-300}$) with interactive Observed vs Expected frequency matrices.
- **Central Limit Theorem (CLT) Monte Carlo Simulator**: Interactive sampling engine demonstrating asymptotic normality as sample size $n$ increases.

### 4. 📊 Model Benchmark & Analytics Lab
- **Multi-Model Leaderboard**:
  - **Random Forest (Tuned)**: **97.32% Accuracy**, **0.9773 F1-Score**
  - **Decision Tree**: 97.24% Accuracy, 0.9766 F1-Score
  - **K-Nearest Neighbors (KNN)**: 95.34% Accuracy, 0.9602 F1-Score
  - **Logistic Regression**: 92.28% Accuracy, 0.9349 F1-Score
- **Magnitude Regressors Benchmark**: Multiple Linear Regression vs Decision Tree Regressor vs Random Forest Regressor metrics ($MAE, MSE, RMSE, R^2$).
- **Interactive Confusion Matrix Inspector** with True/False Positive breakdowns.
- **Feature Importance Decomposition**: Identifies geographic coordinates (Longitude & Latitude ~89%) and Focal Depth (~9%) as dominant drivers of high-magnitude earthquakes.

### 5. 📁 Batch File Diagnostics & Export Studio
- Upload custom sensor telemetry CSV or load the built-in 10-event global test fleet.
- Computes automated magnitude prediction across all 3 regressors, severity tiers, classification, radiated Joules, and TNT equivalent.
- Downloadable annotated CSV report with 1 click.

---

## 🚀 How to Run the Application

### Option A: 1-Click Windows Launcher (Recommended)
Simply double-click the [`run_app.bat`](file:///f:/Eartquake/run_app.bat) file in this directory.

### Option B: Command Line (PowerShell / Terminal)
```bash
py -3.14 -m streamlit run app.py --server.port 8502
```
Then navigate to **`http://localhost:8502`** in your browser.

---

## 📂 Project Structure

```
f:\Eartquake\
├── app.py                                         # Core Streamlit multi-tab application
├── utils.py                                       # Physical formulas, loaders, magnitude tiers, and presets
├── train_models.py                                # Scikit-learn model training & metrics serialization
├── run_app.bat                                    # 1-Click Windows launcher script
├── assets/
│   └── styles.css                                 # Custom dark glassmorphism design system
├── best_earthquake_classifier.joblib              # Tuned Random Forest Classifier (F1: 0.977)
├── earthquake_magnitude_linear_regressor.joblib   # Multiple Linear Regression Magnitude Predictor
├── earthquake_magnitude_dt_regressor.joblib       # Decision Tree Regressor Magnitude Predictor
├── earthquake_magnitude_rf_regressor.joblib       # Random Forest Regressor Magnitude Predictor
├── earthquake_spatial_clusterer.joblib            # 3D Unit-Sphere Spatial K-Means Tectonic Clusterer
├── regional_cluster_models.joblib                 # 8 Dedicated Regional Expert Regressors
├── regional_cluster_metrics.json                  # Precomputed profiles & metrics for all 8 Earth chunks
├── earthquake_data_cleaned.csv                    # Cleaned catalog dataset (48,080 events with cluster labels)
├── model_metrics.json                             # Precomputed benchmarking & evaluation results
├── statistical_inference_results.json             # Precomputed hypothesis testing results
├── sample_earthquake_batch.csv                    # Sample global fleet for batch prediction testing
├── Eartquake_SI_project.ipynb                     # Original ML research Jupyter notebook
└── README.md                                      # Project documentation & user guide
```
