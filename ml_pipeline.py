"""
ml_pipeline.py
ML pipeline callable from the Flask API.
Returns a structured results dict; no files are written to disk.
"""

import io
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE


MODEL_ABBREVS = {
    "Logistic Regression": "Log.Reg.",
    "Decision Tree":       "Dec.Tree",
    "Random Forest":       "Rand.Forest",
    "SVM":                 "SVM",
    "KNN":                 "KNN",
}

# Render free tier has 512 MB RAM — limit SHAP to this many rows
_SHAP_MAX_ROWS = 150


def run_pipeline(file_input) -> dict:
    """
    file_input: str (file path) or io.BytesIO object
    """
    # ── 1. Load ──────────────────────────────────────────────────────────────
    if isinstance(file_input, io.BytesIO):
        file_input.seek(0)
    df = pd.read_csv(file_input)

    # ── 2. Clean ─────────────────────────────────────────────────────────────
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)

    # ── 3. Feature Engineering ───────────────────────────────────────────────
    df['AvgMonthlySpend'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ContractValue']   = df['MonthlyCharges'] * df['tenure']

    # ── 4. Encode ────────────────────────────────────────────────────────────
    df = pd.get_dummies(df, drop_first=True)

    # ── 5. Split features / target ───────────────────────────────────────────
    target_col = [col for col in df.columns if 'Churn' in col][-1]
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # ── 6. SMOTE ─────────────────────────────────────────────────────────────
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # ── 7. Train / test split (70 / 30, then 50/50 → 15/15) ─────────────────
    X_train, X_temp, y_train, y_temp = train_test_split(
        X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled
    )
    _, X_test, _, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )

    # ── 8. Scale ─────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    X_test_df      = pd.DataFrame(X_test_scaled, columns=X.columns)

    # ── 9. Train & evaluate models ───────────────────────────────────────────
    param_grids = {
        "Logistic Regression": {
            "model":  LogisticRegression(max_iter=1000),
            "params": {"C": [0.1, 1, 10]},
        },
        "Decision Tree": {
            "model":  DecisionTreeClassifier(),
            "params": {"max_depth": [5, 10, None]},
        },
        "Random Forest": {
            "model":  RandomForestClassifier(),
            "params": {"n_estimators": [50, 100], "max_depth": [5, 10]},
        },
        "SVM": {
            "model":  SVC(probability=True),
            "params": {"C": [0.1, 1], "kernel": ["linear", "rbf"]},
        },
        "KNN": {
            "model":  KNeighborsClassifier(),
            "params": {"n_neighbors": [3, 5, 7]},
        },
    }

    best_models = {}
    results = []

    for name, config in param_grids.items():
        grid = GridSearchCV(config["model"], config["params"], cv=3, scoring="roc_auc")
        grid.fit(X_train_scaled, y_train)

        model   = grid.best_estimator_
        best_models[name] = model

        y_pred  = model.predict(X_test_scaled)
        y_prob  = model.predict_proba(X_test_scaled)[:, 1]

        results.append({
            "model":     name,
            "accuracy":  round(accuracy_score(y_test, y_pred), 4),
            "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
            "roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
        })

    # ── 10. Best model ───────────────────────────────────────────────────────
    best_result     = max(results, key=lambda r: r["roc_auc"])
    best_model_name = best_result["model"]
    best_model      = best_models[best_model_name]

    y_pred_best = best_model.predict(X_test_scaled)
    y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred_best).ravel()

    churn_predictions = pd.DataFrame({
        "actual":          y_test.values,
        "predicted":       y_pred_best,
        "churn_prob":      y_prob_best,
        "monthly_charges": X_test["MonthlyCharges"].values,
    })

    # ── 11. Feature importance (Random Forest) ───────────────────────────────
    rf_model    = best_models["Random Forest"]
    importances = (
        pd.Series(rf_model.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(10)
    )
    feature_importance = [
        {"feature": feat, "importance": round(float(imp), 4)}
        for feat, imp in importances.items()
    ]

    # ── 12. SHAP (capped at _SHAP_MAX_ROWS to stay within free-tier RAM) ─────
    top_customer_shap  = []
    top_customer_data  = {}
    shap_error         = None

    try:
        import shap

        n_shap = min(_SHAP_MAX_ROWS, len(X_test_df))
        X_shap = X_test_df.iloc[:n_shap]

        explainer   = shap.TreeExplainer(rf_model)
        shap_values = explainer.shap_values(X_shap)

        # Normalise shap_values → shape (n_samples, n_features) for class 1
        if isinstance(shap_values, list):
            shap_class1 = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_class1 = shap_values[:, :, 1]
        else:
            shap_class1 = shap_values

        # Top at-risk customer within the SHAP sample
        shap_probs  = y_prob_best[:n_shap]
        top_idx     = int(np.argmax(shap_probs))
        top_shap_row = shap_class1[top_idx]
        top_feat_vals = X_shap.iloc[top_idx].values

        shap_items = sorted(
            zip(X.columns.tolist(), top_shap_row, top_feat_vals),
            key=lambda x: abs(x[1]),
            reverse=True,
        )[:6]

        top_customer_shap = [
            {
                "feature":       feat,
                "value":         round(float(sv), 4),
                "feature_value": round(float(fv), 4),
                "description":   f"{'Increases' if sv > 0 else 'Decreases'} churn risk",
            }
            for feat, sv, fv in shap_items
        ]

        top_customer_data = {
            "id":              f"C{10000 + top_idx}",
            "monthly_revenue": round(float(churn_predictions.iloc[top_idx]["monthly_charges"]), 2),
            "churn_prob":      round(float(y_prob_best[top_idx]), 2),
            "shap_values":     top_customer_shap,
        }

    except Exception as exc:
        shap_error = str(exc)
        top_customer_data = {
            "id": "N/A", "monthly_revenue": 0.0,
            "churn_prob": 0.0, "shap_values": [],
        }

    # ── 13. Customer risk segmentation ───────────────────────────────────────
    high_risk   = int((y_prob_best > 0.7).sum())
    medium_risk = int(((y_prob_best > 0.4) & (y_prob_best <= 0.7)).sum())
    low_risk    = int((y_prob_best <= 0.4).sum())

    customer_segments = [
        {"name": "High Risk",   "value": high_risk,   "color": "#dc2626"},
        {"name": "Medium Risk", "value": medium_risk, "color": "#f59e0b"},
        {"name": "Low Risk",    "value": low_risk,    "color": "#16a34a"},
    ]

    # ── 14. High-value customers at risk (top 10) ────────────────────────────
    churn_mask          = churn_predictions["predicted"] == 1
    high_value_churners = (
        churn_predictions[churn_mask]
        .sort_values("monthly_charges", ascending=False)
        .head(10)
    )

    high_value_customers = []
    for i, (pos, row) in enumerate(high_value_churners.iterrows()):
        # Use RF feature importance as fallback risk factor when SHAP unavailable
        top_feat_name = importances.index[0]
        if top_customer_shap and pos < len(X_test_df):
            try:
                import shap as _shap  # already imported above if successful
                customer_shap = shap_class1[pos] if pos < len(shap_class1) else None
                if customer_shap is not None:
                    top_feat_name = X.columns[int(np.argmax(np.abs(customer_shap)))]
            except Exception:
                pass
        high_value_customers.append({
            "id":              f"C{10000 + i}",
            "monthly_revenue": round(float(row["monthly_charges"]), 2),
            "churn_prob":      round(float(row["churn_prob"]), 2),
            "risk_factors":    f"Key driver: {top_feat_name}",
        })

    # ── 15. Revenue metrics ───────────────────────────────────────────────────
    churners_df   = churn_predictions[churn_mask]
    total_monthly = float(churners_df["monthly_charges"].sum())
    avg_monthly   = float(churners_df["monthly_charges"].mean()) if len(churners_df) > 0 else 0.0

    revenue_metrics = {
        "estimated_revenue_loss":   round(total_monthly * 12 / 1000, 1),
        "total_likely_churn":       int(churn_mask.sum()),
        "avg_revenue_per_customer": round(avg_monthly, 0),
        "high_value_at_risk":       len(high_value_customers),
    }

    model_comparison_chart = [
        {
            "model":    MODEL_ABBREVS.get(r["model"], r["model"]),
            "accuracy": round(r["accuracy"] * 100, 1),
            "roc_auc":  round(r["roc_auc"]  * 100, 1),
        }
        for r in results
    ]

    return {
        "model_results":          results,
        "best_model_name":        best_model_name,
        "confusion_matrix":       {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "feature_importance":     feature_importance,
        "customer_segments":      customer_segments,
        "high_value_customers":   high_value_customers,
        "revenue_metrics":        revenue_metrics,
        "top_customer_analysis":  top_customer_data,
        "model_comparison_chart": model_comparison_chart,
        **({"shap_error": shap_error} if shap_error else {}),
    }
