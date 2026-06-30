# -*- coding: utf-8 -*-
"""
Streamlit Community Cloud app for disease-specific AKI risk prediction.

Cohorts:
- Arrhythmia
- CAD
- HF
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import xgboost as xgb


# ============================================================
# 0. Basic configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

COHORT_LABELS = {
    "Arrhythmia": "Arrhythmia cohort",
    "CAD": "Coronary artery disease cohort",
    "HF": "Heart failure cohort"
}

COHORT_DESCRIPTIONS = {
    "Arrhythmia": "Prediction model trained in the arrhythmia subgroup.",
    "CAD": "Prediction model trained in the coronary artery disease subgroup.",
    "HF": "Prediction model trained in the heart failure subgroup."
}


# ============================================================
# 1. Load artifacts
# ============================================================

def check_artifact_files(cohort_name):
    cohort_dir = ARTIFACT_DIR / cohort_name

    required_files = [
        cohort_dir / f"{cohort_name}_top10_xgb_model.json",
        cohort_dir / f"{cohort_name}_feature_schema.csv",
        cohort_dir / f"{cohort_name}_metadata.json",
        cohort_dir / f"{cohort_name}_shap_importance_top10.csv",
    ]

    missing_files = [str(p) for p in required_files if not p.exists()]

    if missing_files:
        raise FileNotFoundError(
            "Missing required deployment files:\n" + "\n".join(missing_files)
        )


@st.cache_resource
def load_model(cohort_name):
    check_artifact_files(cohort_name)

    model_path = ARTIFACT_DIR / cohort_name / f"{cohort_name}_top10_xgb_model.json"

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    return booster


@st.cache_data
def load_schema_metadata_shap(cohort_name):
    check_artifact_files(cohort_name)

    cohort_dir = ARTIFACT_DIR / cohort_name

    schema_path = cohort_dir / f"{cohort_name}_feature_schema.csv"
    metadata_path = cohort_dir / f"{cohort_name}_metadata.json"
    shap_path = cohort_dir / f"{cohort_name}_shap_importance_top10.csv"

    schema_df = pd.read_csv(schema_path)
    shap_df = pd.read_csv(shap_path)

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    required_schema_cols = {
        "original_feature",
        "display_feature",
        "input_type",
        "default_value",
        "p01",
        "p99",
        "mean_abs_shap"
    }

    missing_schema_cols = required_schema_cols - set(schema_df.columns)

    if missing_schema_cols:
        raise ValueError(
            f"Feature schema is missing columns: {missing_schema_cols}"
        )

    return schema_df, metadata, shap_df


# ============================================================
# 2. Prediction utilities
# ============================================================

def get_selected_threshold(metadata, threshold_mode):
    if threshold_mode == "High-sensitivity threshold":
        return float(metadata.get("sensitivity80_threshold", 0.5))

    if threshold_mode == "Youden threshold":
        return float(metadata.get("youden_threshold", 0.5))

    return 0.5


def risk_category(probability, threshold):
    if probability >= threshold:
        return "High risk"
    return "Low risk"


def build_input_dataframe(input_values, feature_order):
    X = pd.DataFrame([input_values], columns=feature_order)

    for col in feature_order:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if X.isnull().any().any():
        missing_cols = X.columns[X.isnull().any()].tolist()
        raise ValueError(
            "Input contains missing or non-numeric values in: "
            + ", ".join(missing_cols)
        )

    return X


def predict_one_patient(model, input_values, feature_order, best_iteration):
    X_input = build_input_dataframe(input_values, feature_order)

    dmatrix = xgb.DMatrix(
        X_input,
        feature_names=feature_order
    )

    predict_kwargs = {}

    if best_iteration is not None:
        predict_kwargs["iteration_range"] = (0, int(best_iteration) + 1)

    probability = float(model.predict(dmatrix, **predict_kwargs)[0])

    contributions = model.predict(
        dmatrix,
        pred_contribs=True,
        **predict_kwargs
    )[0]

    feature_contributions = contributions[:-1]

    contrib_df = pd.DataFrame({
        "original_feature": feature_order,
        "contribution_log_odds": feature_contributions
    })

    contrib_df["abs_contribution"] = contrib_df["contribution_log_odds"].abs()

    contrib_df = contrib_df.sort_values(
        "abs_contribution",
        ascending=False
    ).reset_index(drop=True)

    return probability, contrib_df


def predict_batch(model, batch_df, feature_order, best_iteration):
    missing_cols = [c for c in feature_order if c not in batch_df.columns]

    if missing_cols:
        raise ValueError(
            "Uploaded CSV is missing required columns: "
            + ", ".join(missing_cols)
        )

    X_batch = batch_df[feature_order].copy()

    for col in feature_order:
        X_batch[col] = pd.to_numeric(X_batch[col], errors="coerce")

    if X_batch.isnull().any().any():
        missing_summary = X_batch.isnull().sum()
        missing_summary = missing_summary[missing_summary > 0]

        raise ValueError(
            "Uploaded CSV contains missing or non-numeric values:\n"
            + missing_summary.to_string()
        )

    dmatrix = xgb.DMatrix(
        X_batch,
        feature_names=feature_order
    )

    predict_kwargs = {}

    if best_iteration is not None:
        predict_kwargs["iteration_range"] = (0, int(best_iteration) + 1)

    probabilities = model.predict(dmatrix, **predict_kwargs)

    return probabilities


def check_out_of_reference_range(input_values, schema_df):
    warnings = []

    for _, row in schema_df.iterrows():
        feature = row["original_feature"]
        display = row["display_feature"]
        input_type = row["input_type"]

        if input_type == "binary":
            continue

        value = float(input_values[feature])
        p01 = float(row["p01"])
        p99 = float(row["p99"])

        if value < p01 or value > p99:
            warnings.append(
                f"{display}: input value {value:.4f} is outside the training reference range "
                f"P1-P99 [{p01:.4f}, {p99:.4f}]."
            )

    return warnings


# ============================================================
# 3. Page
# ============================================================

st.set_page_config(
    page_title="AKI Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Disease-specific AKI Risk Prediction Using XGBoost")

st.markdown(
    """
This application estimates acute kidney injury risk using disease-specific 
XGBoost models based on the top 10 SHAP-ranked predictors.
"""
)


# ============================================================
# 4. Sidebar
# ============================================================

with st.sidebar:
    st.header("Disease subgroup")

    cohort_name = st.selectbox(
        label="Select disease-specific prediction model",
        options=list(COHORT_LABELS.keys()),
        format_func=lambda x: COHORT_LABELS[x]
    )

    st.caption(COHORT_DESCRIPTIONS[cohort_name])

    st.markdown("---")

    st.header("Risk threshold")

    threshold_mode = st.radio(
        label="Select threshold",
        options=[
            "High-sensitivity threshold",
            "Youden threshold",
            "Fixed 0.5 threshold"
        ],
        index=0
    )

    st.caption(
        "For ICU early warning, a high-sensitivity threshold is usually preferred "
        "to reduce missed high-risk cases."
    )


# ============================================================
# 5. Load selected model
# ============================================================

try:
    schema_df, metadata, shap_df = load_schema_metadata_shap(cohort_name)
    model = load_model(cohort_name)

except Exception as e:
    st.error("Failed to load deployment artifacts.")
    st.exception(e)
    st.stop()


feature_order = schema_df["original_feature"].tolist()
display_map = dict(
    zip(schema_df["original_feature"], schema_df["display_feature"])
)

best_iteration = metadata.get("best_iteration", None)
threshold = get_selected_threshold(metadata, threshold_mode)


tab_predict, tab_batch, tab_model, tab_about = st.tabs(
    [
        "Single-patient prediction",
        "Batch prediction",
        "Model information",
        "Clinical notes"
    ]
)


# ============================================================
# 6. Single-patient prediction
# ============================================================

with tab_predict:
    st.subheader(f"Prediction page: {COHORT_LABELS[cohort_name]}")

    st.info(
        "Enter the values of the disease-specific Top 10 SHAP predictors."
    )

    input_values = {}

    with st.form(key=f"{cohort_name}_single_prediction_form"):
        cols = st.columns(2)

        for i, row in schema_df.iterrows():
            original_feature = str(row["original_feature"])
            display_feature = str(row["display_feature"])
            input_type = str(row["input_type"])
            default_value = float(row["default_value"])
            p01 = float(row["p01"])
            p99 = float(row["p99"])

            help_text = (
                f"Original variable: {original_feature}. "
                f"Training reference interval: P1={p01:.4f}, P99={p99:.4f}."
            )

            with cols[i % 2]:
                if input_type == "binary":
                    default_binary = int(round(default_value))

                    if default_binary not in [0, 1]:
                        default_binary = 0

                    value = st.selectbox(
                        label=display_feature,
                        options=[0, 1],
                        index=default_binary,
                        format_func=lambda x: "Yes / 1" if x == 1 else "No / 0",
                        help=help_text,
                        key=f"{cohort_name}_{original_feature}_single"
                    )

                    input_values[original_feature] = int(value)

                else:
                    value = st.number_input(
                        label=display_feature,
                        value=default_value,
                        format="%.4f",
                        help=help_text,
                        key=f"{cohort_name}_{original_feature}_single"
                    )

                    input_values[original_feature] = float(value)

        submitted = st.form_submit_button("Predict AKI risk")

    if submitted:
        range_warnings = check_out_of_reference_range(
            input_values=input_values,
            schema_df=schema_df
        )

        if range_warnings:
            with st.expander("Input range warnings", expanded=True):
                for msg in range_warnings:
                    st.warning(msg)

        try:
            probability, contrib_df = predict_one_patient(
                model=model,
                input_values=input_values,
                feature_order=feature_order,
                best_iteration=best_iteration
            )

            category = risk_category(probability, threshold)

            st.markdown("---")
            st.subheader("Prediction result")

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    label="Predicted AKI risk",
                    value=f"{probability * 100:.2f}%"
                )

            with col2:
                st.metric(
                    label="Selected threshold",
                    value=f"{threshold * 100:.2f}%"
                )

            with col3:
                st.metric(
                    label="Risk category",
                    value=category
                )

            if category == "High risk":
                st.warning(
                    "The patient is classified as high risk according to the selected threshold."
                )
            else:
                st.success(
                    "The patient is classified as low risk according to the selected threshold."
                )

            st.subheader("Individual feature contribution")

            contrib_df["display_feature"] = contrib_df["original_feature"].map(display_map)

            contrib_show = contrib_df[
                [
                    "display_feature",
                    "original_feature",
                    "contribution_log_odds",
                    "abs_contribution"
                ]
            ].copy()

            contrib_show = contrib_show.rename(
                columns={
                    "display_feature": "Feature",
                    "original_feature": "Original variable",
                    "contribution_log_odds": "Contribution to log-odds",
                    "abs_contribution": "Absolute contribution"
                }
            )

            st.dataframe(contrib_show, use_container_width=True)

            st.caption(
                "Positive contribution increases predicted AKI risk; "
                "negative contribution decreases predicted AKI risk."
            )

            st.subheader("Input summary")

            input_show = pd.DataFrame({
                "Feature": [display_map[f] for f in feature_order],
                "Original variable": feature_order,
                "Input value": [input_values[f] for f in feature_order]
            })

            st.dataframe(input_show, use_container_width=True)

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# ============================================================
# 7. Batch prediction
# ============================================================

with tab_batch:
    st.subheader("Batch prediction")

    st.markdown(
        """
Upload a CSV file containing the required Top 10 original feature columns.
The uploaded file may contain additional columns, but the following columns are required:
"""
    )

    st.code("\n".join(feature_order), language="text")

    template_df = pd.DataFrame(columns=feature_order)

    st.download_button(
        label="Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"{cohort_name}_batch_prediction_template.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload CSV for batch prediction",
        type=["csv"]
    )

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)

            st.write("Uploaded data preview")
            st.dataframe(batch_df.head(), use_container_width=True)

            if st.button("Run batch prediction"):
                probabilities = predict_batch(
                    model=model,
                    batch_df=batch_df,
                    feature_order=feature_order,
                    best_iteration=best_iteration
                )

                result_df = batch_df.copy()
                result_df["predicted_aki_risk"] = probabilities
                result_df["risk_category"] = np.where(
                    result_df["predicted_aki_risk"] >= threshold,
                    "High risk",
                    "Low risk"
                )

                st.success("Batch prediction completed.")
                st.dataframe(result_df, use_container_width=True)

                st.download_button(
                    label="Download prediction results",
                    data=result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{cohort_name}_batch_prediction_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error("Batch prediction failed.")
            st.exception(e)


# ============================================================
# 8. Model information
# ============================================================

with tab_model:
    st.subheader("Model information")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Validation AUROC",
            f"{float(metadata.get('val_auc', np.nan)):.4f}"
        )

    with col2:
        st.metric(
            "Validation AUPRC",
            f"{float(metadata.get('val_auprc', np.nan)):.4f}"
        )

    with col3:
        st.metric(
            "Best iteration",
            f"{metadata.get('best_iteration', 'NA')}"
        )

    with col4:
        st.metric(
            "Current threshold",
            f"{threshold:.4f}"
        )

    st.markdown("### Cohort information")

    cohort_info = {
        "Cohort": metadata.get("cohort_name", cohort_name),
        "Model type": metadata.get("model_type", "XGBoost"),
        "Total rows": metadata.get("n_total_rows", "NA"),
        "Training rows": metadata.get("n_train_rows", "NA"),
        "Validation rows": metadata.get("n_val_rows", "NA"),
        "Training positive rate": metadata.get("train_positive_rate", "NA"),
        "Validation positive rate": metadata.get("val_positive_rate", "NA"),
        "Youden threshold": metadata.get("youden_threshold", "NA"),
        "High-sensitivity threshold": metadata.get("sensitivity80_threshold", "NA"),
    }

    cohort_info_df = pd.DataFrame(
        list(cohort_info.items()),
        columns=["Item", "Value"]
    )

    st.dataframe(cohort_info_df, use_container_width=True)

    st.markdown("### SHAP Top 10 predictors")

    shap_show = shap_df[
        ["feature", "original_feature", "mean_abs_shap"]
    ].copy()

    shap_show = shap_show.rename(
        columns={
            "feature": "Feature",
            "original_feature": "Original variable",
            "mean_abs_shap": "Mean absolute SHAP value"
        }
    )

    st.dataframe(shap_show, use_container_width=True)

    st.markdown("### Feature schema")

    schema_show = schema_df.copy()

    schema_show = schema_show.rename(
        columns={
            "display_feature": "Display feature",
            "original_feature": "Original variable",
            "input_type": "Input type",
            "default_value": "Default value",
            "p01": "P1",
            "p99": "P99",
            "mean_abs_shap": "Mean absolute SHAP value"
        }
    )

    st.dataframe(schema_show, use_container_width=True)


# ============================================================
# 9. Clinical notes
# ============================================================

with tab_about:
    st.subheader("Clinical and methodological notes")

    st.markdown(
        """
1. This application is intended for research demonstration and model interpretation.
2. The model output is an estimated probability of AKI risk, not a standalone clinical diagnosis.
3. Three disease-specific models are deployed:
   - Arrhythmia cohort
   - Coronary artery disease cohort
   - Heart failure cohort
4. Each prediction page uses the corresponding disease-specific Top 10 SHAP predictors.
5. The disease setting is a model-selection step, not a mutually exclusive diagnostic classifier.
6. The threshold should be fixed before external validation or prospective clinical evaluation.
7. Before real clinical use, the model requires external validation, calibration assessment, usability testing, and clinical workflow evaluation.
"""
    )