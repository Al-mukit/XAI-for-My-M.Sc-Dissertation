"""
backend.py
==========
All non-UI logic for the XAI-SOC Streamlit dashboard. Loads artifacts
exactly as produced by the final pipeline notebook's Section 13
("Generate Backend Artifact Files"):

    artifacts/
        best_model.joblib   (or best_model_mlp.keras, if the MLP won)
        scaler.joblib
        label_encoder.joblib
        background_sample.csv
        metadata.json

metadata.json contains: best_model_name, best_model_file,
best_model_is_tree_based, selection_metric, feature_names, classes,
metrics, all_models_comparison, sample_frac, n_train_rows, n_test_rows,
datasets.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
from lime.lime_tabular import LimeTabularExplainer

ARTIFACT_DIR = os.environ.get("XAI_SOC_ARTIFACT_DIR", "artifacts")

REQUIRED_FILES = [
    "scaler.joblib",
    "label_encoder.joblib",
    "background_sample.csv",
    "metadata.json",
]


def artifacts_missing(artifact_dir=ARTIFACT_DIR):
    missing = [f for f in REQUIRED_FILES if not os.path.exists(os.path.join(artifact_dir, f))]
    # the model file itself has a variable name/extension (.joblib or .keras)
    # depending on which model won, so it's checked separately once metadata
    # is readable
    meta_path = os.path.join(artifact_dir, "metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
        model_file = meta.get("best_model_file", "best_model.joblib")
        if not os.path.exists(os.path.join(artifact_dir, model_file)):
            missing.append(model_file)
    return missing


def _load_model(artifact_dir, meta):
    model_file = meta.get("best_model_file", "best_model.joblib")
    model_path = os.path.join(artifact_dir, model_file)

    if model_file.endswith(".keras") or model_file.endswith(".h5"):
        from tensorflow.keras.models import load_model as keras_load_model
        return keras_load_model(model_path), True
    return joblib.load(model_path), False


@st.cache_resource(show_spinner="Loading trained model and explainers...")
def load_artifacts(artifact_dir=ARTIFACT_DIR):
    with open(os.path.join(artifact_dir, "metadata.json")) as f:
        meta = json.load(f)

    model, is_keras = _load_model(artifact_dir, meta)
    scaler = joblib.load(os.path.join(artifact_dir, "scaler.joblib"))
    label_encoder = joblib.load(os.path.join(artifact_dir, "label_encoder.joblib"))
    background = pd.read_csv(os.path.join(artifact_dir, "background_sample.csv"))

    feature_names = meta["feature_names"]
    is_tree_based = meta.get("best_model_is_tree_based", False)

    if is_tree_based:
        shap_explainer = shap.TreeExplainer(model)
        shap_mode = "tree"
    else:
        # Generic explainer: slower, so background is kept small. See the
        # notebook's own note about KNN/MLP being expensive to explain.
        bg_for_shap = shap.sample(
            background[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0),
            min(30, len(background)),
            random_state=42,
        )
        predict_fn = _make_predict_fn(model, is_keras, feature_names)
        shap_explainer = shap.Explainer(predict_fn, bg_for_shap)
        shap_mode = "general"

    lime_explainer = LimeTabularExplainer(
        training_data=background[feature_names].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy(),
        feature_names=feature_names,
        class_names=meta["classes"],
        mode="classification",
        discretize_continuous=True,
        random_state=42,
    )

    return {
        "model": model,
        "is_keras": is_keras,
        "is_tree_based": is_tree_based,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "meta": meta,
        "background": background,
        "shap_explainer": shap_explainer,
        "shap_mode": shap_mode,
        "lime_explainer": lime_explainer,
    }


def _make_predict_fn(model, is_keras, feature_names):
    """Unified predict_proba-style function for both sklearn/XGBoost and Keras models."""
    if is_keras:
        return lambda arr: model.predict(np.asarray(arr, dtype=float), verbose=0)
    return lambda arr: model.predict_proba(pd.DataFrame(np.asarray(arr, dtype=float), columns=feature_names))


def predict_alert(artifacts, raw_row: pd.Series):
    """raw_row: a Series of RAW (unscaled) feature values for one flow.
    Returns (predicted_class_name, class_probabilities_dict, scaled_row_array)."""
    feature_names = artifacts["meta"]["feature_names"]
    x = raw_row[feature_names].to_frame().T.apply(pd.to_numeric, errors="coerce").fillna(0)
    x_scaled = artifacts["scaler"].transform(x)
    x_scaled_df = pd.DataFrame(x_scaled, columns=feature_names)

    predict_fn = _make_predict_fn(artifacts["model"], artifacts["is_keras"], feature_names)
    proba = predict_fn(x_scaled_df.to_numpy())[0]

    classes = artifacts["label_encoder"].classes_
    pred_idx = int(np.argmax(proba))
    proba_dict = {cls: float(p) for cls, p in zip(classes, proba)}
    return classes[pred_idx], proba_dict, x_scaled[0]


def get_shap_explanation(artifacts, x_scaled_row, class_name):
    """Return a shap.Explanation object for one instance and one class."""
    feature_names = artifacts["meta"]["feature_names"]
    classes = list(artifacts["label_encoder"].classes_)
    class_idx = classes.index(class_name)

    row_df = pd.DataFrame([x_scaled_row], columns=feature_names)
    exp = artifacts["shap_explainer"](row_df)
    if exp.values.ndim == 3:
        return exp[0, :, class_idx]
    return exp[0]


def get_lime_explanation(artifacts, x_scaled_row, num_features=10):
    """Return a LIME Explanation object for one instance (predicted class)."""
    feature_names = artifacts["meta"]["feature_names"]
    predict_fn = _make_predict_fn(artifacts["model"], artifacts["is_keras"], feature_names)
    lime_explainer = artifacts["lime_explainer"]
    return lime_explainer.explain_instance(
        data_row=np.asarray(x_scaled_row, dtype=float),
        predict_fn=predict_fn,
        num_features=num_features,
    )


def get_global_shap_summary(artifacts, class_name, sample_size=200, random_state=42):
    """Compute a global SHAP summary (feature importance) for one class,
    sampled from the background set for tractability."""
    feature_names = artifacts["meta"]["feature_names"]
    classes = list(artifacts["label_encoder"].classes_)
    class_idx = classes.index(class_name)

    # non-tree explainers are expensive per-call; cap the sample harder
    cap = sample_size if artifacts["shap_mode"] == "tree" else min(sample_size, 30)

    bg = artifacts["background"][feature_names].apply(pd.to_numeric, errors="coerce").fillna(0)
    sample = bg.sample(n=min(cap, len(bg)), random_state=random_state)
    exp = artifacts["shap_explainer"](sample)
    if exp.values.ndim == 3:
        return exp.values[:, :, class_idx], sample
    return exp.values, sample
