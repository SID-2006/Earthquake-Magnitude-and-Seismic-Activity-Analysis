import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
)

CLUSTER_METADATA = {
    0: {
        "name": "NW Pacific / Kuril-Kamchatka-Japan Arc",
        "description": "High-velocity subduction zone responsible for massive megathrust events.",
        "tectonic_type": "Subduction Megathrust"
    },
    1: {
        "name": "Caribbean Basin & Antilles Fault System",
        "description": "Complex strike-slip and subduction boundary with moderate to severe seismicity.",
        "tectonic_type": "Transform / Oceanic Trench"
    },
    2: {
        "name": "SE Asia / Sunda-Banda & Coral Triangle",
        "description": "Indo-Australian & Sunda plate convergence, high frequency of tsunami-generating earthquakes.",
        "tectonic_type": "Subduction Zone"
    },
    3: {
        "name": "Alpine-Himalayan & Mediterranean Belt",
        "description": "Continental collision zone (Eurasian-African-Indian plates) with shallow destructive ruptures.",
        "tectonic_type": "Continental Collision"
    },
    4: {
        "name": "North America Continental & San Andreas",
        "description": "Continental transform boundary and Basin & Range extensional province; shallow crustal events.",
        "tectonic_type": "Continental Transform / Extensional"
    },
    5: {
        "name": "SW Pacific / Tonga-Kermadec-Fiji Trench",
        "description": "Deepest Wadati-Benioff subduction zone on Earth with intense deep-mantle seismicity.",
        "tectonic_type": "Deep Ocean Trench"
    },
    6: {
        "name": "South America / Andean Subduction & Chile",
        "description": "Nazca plate subducting under South American plate; historically produces the largest earthquakes.",
        "tectonic_type": "Oceanic-Continental Subduction"
    },
    7: {
        "name": "North Pacific / Alaska-Aleutians & Hawaii",
        "description": "Aleutian megathrust subduction arc and Hawaiian volcanic hotspot seismicity.",
        "tectonic_type": "Subduction Arc & Volcanic Hotspot"
    }
}

def train_and_export():
    print("Loading earthquake_data_cleaned.csv...")
    df = pd.read_csv("earthquake_data_cleaned.csv")
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    features = ["depth", "latitude", "longitude", "month", "hour"]
    
    # Ensure high_magnitude is defined
    if "high_magnitude" not in df.columns:
        df["high_magnitude"] = (df["mag"] >= 4.0).astype(int)

    # 1. Spherical 3D Coordinates for Tectonic Clustering
    print("Computing 3D spherical coordinates and clustering tectonic chunks...")
    lat_rad = np.radians(df["latitude"])
    lon_rad = np.radians(df["longitude"])
    df["cart_x"] = np.cos(lat_rad) * np.cos(lon_rad)
    df["cart_y"] = np.cos(lat_rad) * np.sin(lon_rad)
    df["cart_z"] = np.sin(lat_rad)

    # Spatial clusterer (K-Means on unit sphere)
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    df["cluster_id"] = kmeans.fit_predict(df[["cart_x", "cart_y", "cart_z"]])
    df["cluster_name"] = df["cluster_id"].map(lambda cid: CLUSTER_METADATA[cid]["name"])

    # Export augmented dataset with cluster info
    df.to_csv("earthquake_data_cleaned.csv", index=False)
    print("Updated earthquake_data_cleaned.csv with cluster_id & cluster_name columns.")

    # Save spatial clusterer
    joblib.dump(kmeans, "earthquake_spatial_clusterer.joblib")
    print("Saved earthquake_spatial_clusterer.joblib successfully!")

    # Clean data subset
    sub = df[features + ["mag", "high_magnitude", "cluster_id"]].dropna().copy()
    X = sub[features]
    y_cls = sub["high_magnitude"].astype(int)
    y_reg = sub["mag"]

    # Aligned Stratified train/test split
    indices = np.arange(len(sub))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.2, random_state=42, stratify=y_cls
    )

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_cls.iloc[train_idx], y_cls.iloc[test_idx]
    yr_train, yr_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
    sub_train, sub_test = sub.iloc[train_idx], sub.iloc[test_idx]

    print(f"Training set: {len(X_train)} samples | Testing set: {len(X_test)} samples")

    # 2. Best Classification Model: Random Forest
    print("Training Best Random Forest Classifier...")
    rf_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ))
    ])
    rf_pipeline.fit(X_train, y_train)
    rf_pred = rf_pipeline.predict(X_test)
    joblib.dump(rf_pipeline, "best_earthquake_classifier.joblib")
    print("Saved best_earthquake_classifier.joblib successfully!")

    # 3. Global Regression Models (with aligned targets)
    print("Training Multiple Linear Regression on Magnitude...")
    linear_reg_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", LinearRegression())
    ])
    linear_reg_pipeline.fit(X_train, yr_train)
    lr_pred = linear_reg_pipeline.predict(X_test)
    joblib.dump(linear_reg_pipeline, "earthquake_magnitude_linear_regressor.joblib")

    lr_mae = float(mean_absolute_error(yr_test, lr_pred))
    lr_mse = float(mean_squared_error(yr_test, lr_pred))
    lr_rmse = float(np.sqrt(lr_mse))
    lr_r2 = float(r2_score(yr_test, lr_pred))
    print(f"-> Linear Regression: MAE={lr_mae:.4f}, R2={lr_r2:.4f}")

    print("Training Decision Tree Regressor on Magnitude...")
    dt_reg_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", DecisionTreeRegressor(max_depth=10, min_samples_leaf=5, random_state=42))
    ])
    dt_reg_pipeline.fit(X_train, yr_train)
    dt_pred = dt_reg_pipeline.predict(X_test)
    joblib.dump(dt_reg_pipeline, "earthquake_magnitude_dt_regressor.joblib")

    dt_mae = float(mean_absolute_error(yr_test, dt_pred))
    dt_mse = float(mean_squared_error(yr_test, dt_pred))
    dt_rmse = float(np.sqrt(dt_mse))
    dt_r2 = float(r2_score(yr_test, dt_pred))
    print(f"-> Decision Tree Regressor: MAE={dt_mae:.4f}, R2={dt_r2:.4f}")

    print("Training Random Forest Regressor on Magnitude...")
    rf_reg_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("regressor", RandomForestRegressor(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1))
    ])
    rf_reg_pipeline.fit(X_train, yr_train)
    rf_reg_pred = rf_reg_pipeline.predict(X_test)
    joblib.dump(rf_reg_pipeline, "earthquake_magnitude_regressor.joblib")
    joblib.dump(rf_reg_pipeline, "earthquake_magnitude_rf_regressor.joblib")

    rf_mae = float(mean_absolute_error(yr_test, rf_reg_pred))
    rf_mse = float(mean_squared_error(yr_test, rf_reg_pred))
    rf_rmse = float(np.sqrt(rf_mse))
    rf_r2 = float(r2_score(yr_test, rf_reg_pred))
    print(f"-> Random Forest Regressor: MAE={rf_mae:.4f}, R2={rf_r2:.4f}")

    # 4. Regional Cluster-Wise Models (Specialized local experts per earth chunk)
    print("Training 8 Regional Area-Wise Specialized Regressors...")
    regional_models = {}
    regional_metrics = {}
    cluster_mixture_preds = np.zeros(len(X_test))

    for cid in range(8):
        c_meta = CLUSTER_METADATA[cid]
        c_name = c_meta["name"]
        
        # Local training data
        c_tr_mask = (sub_train["cluster_id"] == cid)
        c_te_mask = (sub_test["cluster_id"] == cid)
        c_all_df = df[df["cluster_id"] == cid]
        
        local_X_tr = X_train[c_tr_mask]
        local_yr_tr = yr_train[c_tr_mask]
        local_X_te = X_test[c_te_mask]
        local_yr_te = yr_test[c_te_mask]
        
        # Train localized regressor
        local_reg = RandomForestRegressor(
            n_estimators=60,
            max_depth=10,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        local_reg.fit(local_X_tr, local_yr_tr)
        
        # Evaluate local regressor
        local_pred = local_reg.predict(local_X_te)
        cluster_mixture_preds[c_te_mask] = local_pred
        
        l_mae = float(mean_absolute_error(local_yr_te, local_pred))
        l_mse = float(mean_squared_error(local_yr_te, local_pred))
        l_rmse = float(np.sqrt(l_mse))
        l_r2 = float(r2_score(local_yr_te, local_pred))
        
        # Local feature importances
        local_imp = local_reg.feature_importances_
        local_feat_imp = [{"Feature": f, "Importance": round(float(imp), 4)} for f, imp in zip(features, local_imp)]
        local_feat_imp.sort(key=lambda x: x["Importance"], reverse=True)
        
        # Centroid coordinates
        center_cart = kmeans.cluster_centers_[cid]
        c_lat = float(np.degrees(np.arcsin(center_cart[2])))
        c_lon = float(np.degrees(np.arctan2(center_cart[1], center_cart[0])))
        
        # Top geographical places
        places = c_all_df["place"].dropna().astype(str)
        top_places = places.apply(lambda p: p.split(",")[-1].strip() if "," in p else p.strip()).value_counts().head(4).to_dict()
        
        regional_models[str(cid)] = local_reg
        regional_metrics[str(cid)] = {
            "cluster_id": cid,
            "name": c_name,
            "description": c_meta["description"],
            "tectonic_type": c_meta["tectonic_type"],
            "centroid": {"latitude": round(c_lat, 2), "longitude": round(c_lon, 2)},
            "total_records": int(len(c_all_df)),
            "percentage_of_catalog": round(float(len(c_all_df) / len(df) * 100), 2),
            "mean_mag": round(float(c_all_df["mag"].mean()), 2),
            "max_mag": round(float(c_all_df["mag"].max()), 2),
            "p90_mag": round(float(c_all_df["mag"].quantile(0.90)), 2),
            "high_mag_ratio": round(float((c_all_df["mag"] >= 4.0).mean()), 4),
            "severe_mag_ratio": round(float((c_all_df["mag"] >= 6.0).mean()), 4),
            "mean_depth": round(float(c_all_df["depth"].mean()), 2),
            "max_depth": round(float(c_all_df["depth"].max()), 2),
            "top_regions": top_places,
            "mae": round(l_mae, 4),
            "mse": round(l_mse, 4),
            "rmse": round(l_rmse, 4),
            "r2": round(l_r2, 4),
            "feature_importances": local_feat_imp
        }
        print(f"  -> Cluster {cid} [{c_name[:25]}]: MAE={l_mae:.4f}, Mean Mag={c_all_df['mag'].mean():.2f}, Max Mag={c_all_df['mag'].max():.2f}")

    joblib.dump(regional_models, "regional_cluster_models.joblib")
    print("Saved regional_cluster_models.joblib successfully!")

    with open("regional_cluster_metrics.json", "w", encoding="utf-8") as f:
        json.dump(regional_metrics, f, indent=2)
    print("Saved regional_cluster_metrics.json successfully!")

    mix_mae = float(mean_absolute_error(yr_test, cluster_mixture_preds))
    mix_mse = float(mean_squared_error(yr_test, cluster_mixture_preds))
    mix_rmse = float(np.sqrt(mix_mse))
    mix_r2 = float(r2_score(yr_test, cluster_mixture_preds))
    print(f"Overall Regional Mixture Regressor: MAE={mix_mae:.4f}, RMSE={mix_rmse:.4f}, R2={mix_r2:.4f}")

    # 5. Baseline Classifiers benchmark
    print("Evaluating baseline models for comparison hub...")
    classifiers = {
        "Random Forest (Tuned)": rf_pipeline,
        "Logistic Regression": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
        ]),
        "Decision Tree": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42))
        ]),
        "K-Nearest Neighbors": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=7))
        ])
    }

    comparison_results = []
    confusion_matrices = {}

    for name, model in classifiers.items():
        if name != "Random Forest (Tuned)":
            model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, pred)
        prec = precision_score(y_test, pred, zero_division=0)
        rec = recall_score(y_test, pred, zero_division=0)
        f1 = f1_score(y_test, pred, zero_division=0)
        cm = confusion_matrix(y_test, pred).tolist()
        confusion_matrices[name] = cm

        comparison_results.append({
            "Model": name,
            "Accuracy": round(float(acc), 4),
            "Precision": round(float(prec), 4),
            "Recall": round(float(rec), 4),
            "F1_Score": round(float(f1), 4)
        })
        print(f"-> {name}: Accuracy={acc:.4f}, F1={f1:.4f}")

    # Global feature importances
    rf_clf = rf_pipeline.named_steps["classifier"]
    importances = rf_clf.feature_importances_
    feat_imp = [{"Feature": f, "Importance": round(float(imp), 4)} for f, imp in zip(features, importances)]
    feat_imp.sort(key=lambda x: x["Importance"], reverse=True)

    rf_reg = rf_reg_pipeline.named_steps["regressor"]
    reg_importances = rf_reg.feature_importances_
    reg_feat_imp = [{"Feature": f, "Importance": round(float(imp), 4)} for f, imp in zip(features, reg_importances)]
    reg_feat_imp.sort(key=lambda x: x["Importance"], reverse=True)

    dt_reg = dt_reg_pipeline.named_steps["regressor"]
    dt_importances = dt_reg.feature_importances_
    dt_feat_imp = [{"Feature": f, "Importance": round(float(imp), 4)} for f, imp in zip(features, dt_importances)]
    dt_feat_imp.sort(key=lambda x: x["Importance"], reverse=True)

    metrics = {
        "classifiers_comparison": comparison_results,
        "confusion_matrices": confusion_matrices,
        "feature_importances": feat_imp,
        "regressor_feature_importances": reg_feat_imp,
        "dt_feature_importances": dt_feat_imp,
        "regression_models": {
            "Multiple Linear Regression": {
                "MAE": round(lr_mae, 4),
                "MSE": round(lr_mse, 4),
                "RMSE": round(lr_rmse, 4),
                "R2": round(lr_r2, 4)
            },
            "Decision Tree Regressor": {
                "MAE": round(dt_mae, 4),
                "MSE": round(dt_mse, 4),
                "RMSE": round(dt_rmse, 4),
                "R2": round(dt_r2, 4)
            },
            "Random Forest Regressor (Global)": {
                "MAE": round(rf_mae, 4),
                "MSE": round(rf_mse, 4),
                "RMSE": round(rf_rmse, 4),
                "R2": round(rf_r2, 4)
            },
            "Regional Area-Wise Mixture Regressor": {
                "MAE": round(mix_mae, 4),
                "MSE": round(mix_mse, 4),
                "RMSE": round(mix_rmse, 4),
                "R2": round(mix_r2, 4)
            }
        },
        "regression_metrics": {
            "MAE": round(rf_mae, 4),
            "MSE": round(rf_mse, 4),
            "RMSE": round(rf_rmse, 4),
            "R2": round(rf_r2, 4)
        },
        "regional_clusters": regional_metrics,
        "dataset_stats": {
            "total_records": len(df),
            "shallow_count": int((df["depth"] <= 70).sum()),
            "intermediate_count": int(((df["depth"] > 70) & (df["depth"] <= 300)).sum()),
            "deep_count": int((df["depth"] > 300).sum()),
            "high_mag_count": int((df["mag"] >= 4.0).sum()),
            "low_mag_count": int((df["mag"] < 4.0).sum()),
            "mean_mag": round(float(df["mag"].mean()), 2),
            "max_mag": round(float(df["mag"].max()), 2),
            "min_mag": round(float(df["mag"].min()), 2),
            "mean_depth": round(float(df["depth"].mean()), 2),
            "max_depth": round(float(df["depth"].max()), 2)
        }
    }

    with open("model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model_metrics.json successfully!")
    print("Training pipeline finished successfully.")

if __name__ == "__main__":
    train_and_export()
