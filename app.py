# -*- coding: utf-8 -*-
"""
Streamlit Community Cloud app for CRS-AKI risk prediction.

Cohorts:
- Arrhythmia
- CAD
- HF

Manuscript-oriented interface:
- compact title
- single-patient and batch prediction only
- no model information tab
- no clinical notes tab
- no individual contribution table
- no input summary table
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
    "Arrhythmia": "Arrhythmia",
    "CAD": "Coronary artery disease",
    "HF": "Heart failure"
}

COHORT_DESCRIPTIONS = {
    "Arrhythmia": "Arrhythmia subgroup model",
    "CAD": "Coronary artery disease subgroup model",
    "HF": "Heart failure subgroup model"
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
def load_schema_metadata(cohort_name):
    check_artifact_files(cohort_name)

    cohort_dir = ARTIFACT_DIR / cohort_name

    schema_path = cohort_dir / f"{cohort_name}_feature_schema.csv"
    metadata_path = cohort_dir / f"{cohort_name}_metadata.json"

    schema_df = pd.read_csv(schema_path)

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

    return schema_df, metadata


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

    return probability


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
                f"{display}: {value:.4f} is outside the training reference interval "
                f"[{p01:.4f}, {p99:.4f}]."
            )

    return warnings


# ============================================================
# 3. Page setup and manuscript-oriented CSS
# ============================================================

st.set_page_config(
    page_title="CRS-AKI Risk Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    html, body, [class*="css"] {
        font-size: 18px !important;
    }

    .stApp {
        font-size: 18px !important;
    }

    label, p, span, div, button, input, textarea {
        font-size: 1.02rem;
    }

    .block-container {
        padding-top: 2.0rem;
        padding-bottom: 2.0rem;
        max-width: 1460px;
        padding-left: 2.2rem;
        padding-right: 2.2rem;
    }

    h1 {
        font-size: 2.85rem !important;
        font-weight: 750 !important;
        letter-spacing: -0.02em;
        color: #111827;
        margin-bottom: 0.35rem !important;
    }

    h2, h3 {
        color: #111827;
        letter-spacing: -0.01em;
    }

    section[data-testid="stSidebar"] {
        min-width: 330px !important;
        width: 330px !important;
    }

    section[data-testid="stSidebar"] > div {
        width: 330px !important;
    }

    div[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E5E7EB;
    }

    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        font-size: 1.48rem !important;
        font-weight: 760 !important;
        color: #111827 !important;
    }

    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] p,
    div[data-testid="stSidebar"] span {
        font-size: 1.15rem !important;
        line-height: 1.45 !important;
    }

    div[data-testid="stSidebar"] [data-baseweb="select"] {
        font-size: 1.15rem !important;
    }

    div[data-testid="stSidebar"] [role="radiogroup"] label {
        margin-bottom: 0.38rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1.1rem;
        border-bottom: 1px solid #E5E7EB;
    }

    .stTabs [data-baseweb="tab"] {
        font-size: 1.10rem;
        font-weight: 600;
        padding: 0.65rem 0.25rem;
    }

    div[data-testid="stForm"] {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 1.15rem 1.25rem 0.9rem 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
        width: 100%;
    }

    div[data-testid="stFormSubmitButton"] button {
        background-color: #0F766E;
        border: 1px solid #0F766E;
        color: white;
        font-size: 1.08rem !important;
        font-weight: 720;
        border-radius: 8px;
        padding: 0.62rem 1.35rem;
    }

    div[data-testid="stFormSubmitButton"] button:hover {
        background-color: #115E59;
        border: 1px solid #115E59;
        color: white;
    }

    .section-title {
        font-size: 1.42rem;
        font-weight: 720;
        margin: 0.25rem 0 0.85rem 0;
        color: #111827;
    }

    .model-tag {
        display: inline-block;
        background: #ECFDF5;
        color: #065F46;
        border: 1px solid #A7F3D0;
        border-radius: 999px;
        padding: 0.42rem 0.85rem;
        font-size: 1.02rem;
        font-weight: 700;
        margin-bottom: 1.05rem;
    }

    .result-panel {
        margin-top: 1.4rem;
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.055);
    }

    .result-title {
        font-size: 1.42rem;
        font-weight: 750;
        color: #111827;
        margin-bottom: 1.0rem;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 1rem;
    }

    .metric-card {
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        background: #F9FAFB;
        padding: 1.05rem 1.05rem 0.95rem 1.05rem;
    }

    .metric-label {
        font-size: 0.98rem;
        color: #6B7280;
        font-weight: 700;
        margin-bottom: 0.45rem;
        text-transform: uppercase;
        letter-spacing: 0.03em;
    }

    .metric-value {
        font-size: 2.32rem;
        line-height: 1.1;
        font-weight: 780;
        color: #111827;
    }

    .risk-high {
        color: #B91C1C !important;
    }

    .risk-low {
        color: #047857 !important;
    }

    .result-note-high {
        margin-top: 1.0rem;
        border-left: 4px solid #DC2626;
        background: #FEF2F2;
        color: #7F1D1D;
        padding: 0.75rem 0.9rem;
        border-radius: 10px;
        font-weight: 650;
    }

    .result-note-low {
        margin-top: 1.0rem;
        border-left: 4px solid #059669;
        background: #ECFDF5;
        color: #064E3B;
        padding: 0.75rem 0.9rem;
        border-radius: 10px;
        font-weight: 650;
    }

    .batch-card {
        background: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 14px;
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.045);
    }

    div[data-testid="stWidgetLabel"] label,
    div[data-testid="stWidgetLabel"] p {
        font-size: 1.06rem !important;
        font-weight: 650 !important;
        color: #111827 !important;
    }

    div[data-baseweb="input"] input,
    div[data-baseweb="select"] div {
        font-size: 1.05rem !important;
    }

    div[data-testid="stNumberInput"] input {
        font-size: 1.05rem !important;
    }

    .result-note-high,
    .result-note-low {
        font-size: 1.05rem !important;
    }

</style>
""",
    unsafe_allow_html=True
)

st.title("CRS-AKI Risk Prediction")


# ============================================================
# 4. Sidebar
# ============================================================

with st.sidebar:
    st.header("Model settings")

    cohort_name = st.selectbox(
        label="Disease subgroup",
        options=list(COHORT_LABELS.keys()),
        format_func=lambda x: COHORT_LABELS[x]
    )

    st.markdown(
        f"""
        <div class="model-tag">{COHORT_DESCRIPTIONS[cohort_name]}</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    threshold_mode = st.radio(
        label="Risk threshold",
        options=[
            "High-sensitivity threshold",
            "Youden threshold",
            "Fixed 0.5 threshold"
        ],
        index=0
    )


# ============================================================
# 5. Load selected model
# ============================================================

try:
    schema_df, metadata = load_schema_metadata(cohort_name)
    model = load_model(cohort_name)

except Exception as e:
    st.error("Failed to load deployment artifacts.")
    st.exception(e)
    st.stop()


feature_order = schema_df["original_feature"].tolist()
best_iteration = metadata.get("best_iteration", None)
threshold = get_selected_threshold(metadata, threshold_mode)


tab_predict, tab_batch = st.tabs(
    [
        "Single-patient prediction",
        "Batch prediction"
    ]
)


# ============================================================
# 6. Single-patient prediction
# ============================================================

with tab_predict:
    st.markdown(
        f"""
        <div class="section-title">Prediction page: {COHORT_LABELS[cohort_name]}</div>
        """,
        unsafe_allow_html=True
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
                f"Reference interval: P1={p01:.4f}, P99={p99:.4f}."
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

        submitted = st.form_submit_button("Predict CRS-AKI Risk")

    if submitted:
        range_warnings = check_out_of_reference_range(
            input_values=input_values,
            schema_df=schema_df
        )

        if range_warnings:
            with st.expander("Input range warnings", expanded=False):
                for msg in range_warnings:
                    st.warning(msg)

        try:
            probability = predict_one_patient(
                model=model,
                input_values=input_values,
                feature_order=feature_order,
                best_iteration=best_iteration
            )

            category = risk_category(probability, threshold)
            risk_class = "risk-high" if category == "High risk" else "risk-low"
            note_class = "result-note-high" if category == "High risk" else "result-note-low"
            note_text = (
                "The predicted risk exceeds the selected decision threshold."
                if category == "High risk"
                else "The predicted risk is below the selected decision threshold."
            )

            st.markdown(
                f"""
                <div class="result-panel">
                    <div class="result-title">Prediction result</div>
                    <div class="metric-grid">
                        <div class="metric-card">
                            <div class="metric-label">Predicted CRS-AKI risk</div>
                            <div class="metric-value">{probability * 100:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Selected threshold</div>
                            <div class="metric-value">{threshold * 100:.2f}%</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">Risk category</div>
                            <div class="metric-value {risk_class}">{category}</div>
                        </div>
                    </div>
                    <div class="{note_class}">{note_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error("Prediction failed.")
            st.exception(e)


# ============================================================
# 7. Batch prediction
# ============================================================

with tab_batch:
    st.markdown(
        """
        <div class="section-title">Batch prediction</div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="batch-card">
        Upload a CSV file containing the required original feature columns for the selected subgroup.
        </div>
        """,
        unsafe_allow_html=True
    )

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

            st.dataframe(
                batch_df.head(),
                use_container_width=True
            )

            if st.button("Run batch prediction"):
                probabilities = predict_batch(
                    model=model,
                    batch_df=batch_df,
                    feature_order=feature_order,
                    best_iteration=best_iteration
                )

                result_df = batch_df.copy()
                result_df["predicted_crs_aki_risk"] = probabilities
                result_df["risk_category"] = np.where(
                    result_df["predicted_crs_aki_risk"] >= threshold,
                    "High risk",
                    "Low risk"
                )

                st.success("Batch prediction completed.")

                st.dataframe(
                    result_df,
                    use_container_width=True
                )

                st.download_button(
                    label="Download prediction results",
                    data=result_df.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"{cohort_name}_batch_prediction_results.csv",
                    mime="text/csv"
                )

        except Exception as e:
            st.error("Batch prediction failed.")
            st.exception(e)
