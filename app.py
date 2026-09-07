"""
XAI-SOC Streamlit Dashboard
============================
Analyst-facing interface fulfilling proposal Objective 3 / Deliverable
"Web app using Streamlit that displays alerts, predictions and
visualizations with SHAP/LIME".

Run with:
    streamlit run app.py

Expects the trained artifacts produced by the final pipeline notebook
(Section 13, "Generate Backend Artifact Files") in the `artifacts/`
directory next to this file (override with the XAI_SOC_ARTIFACT_DIR env
var): best_model.joblib (or best_model_mlp.keras), scaler.joblib,
label_encoder.joblib, background_sample.csv, metadata.json
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st

from modules.backend import (
    artifacts_missing,
    get_global_shap_summary,
    get_lime_explanation,
    get_shap_explanation,
    load_artifacts,
    predict_alert,
    ARTIFACT_DIR,
)

st.set_page_config(
    page_title="XAI-SOC | Alert Triage Dashboard",
    page_icon="🛡️",
    layout="wide",
)


def main():
    st.title("🛡️ XAI-SOC: Explainable Alert Triage")
    st.caption(
        "Best-performing model (auto-selected) with SHAP + LIME explanations, "
        "for SOC analyst decision support (CICIDS2017/2018)."
    )

    missing = artifacts_missing(ARTIFACT_DIR)
    if missing:
        st.error(
            f"Trained artifacts not found/incomplete in `{ARTIFACT_DIR}/`. "
            f"Missing: {', '.join(missing)}.\n\n"
            "Make sure `artifacts/` sits next to `app.py` (or set the "
            "`XAI_SOC_ARTIFACT_DIR` environment variable), and that it "
            "contains the files produced by the notebook's artifact-export "
            "step: best_model.joblib (or best_model_mlp.keras), "
            "scaler.joblib, label_encoder.joblib, background_sample.csv, "
            "metadata.json."
        )
        st.stop()

    artifacts = load_artifacts(ARTIFACT_DIR)
    meta = artifacts["meta"]

    st.sidebar.header("Model in use")
    st.sidebar.markdown(f"**{meta.get('best_model_name', 'Unknown')}**")
    st.sidebar.caption(
        f"Selected by {meta.get('selection_metric', 'f1_macro')} "
        f"across {len(meta.get('all_models_comparison', []))} candidate models."
    )
    m = meta["metrics"]
    st.sidebar.metric("Test accuracy", f"{m['accuracy']*100:.2f}%")
    st.sidebar.metric("Test macro F1", f"{m['f1_macro']:.3f}")

    tab_triage, tab_global, tab_model = st.tabs(
        ["🔎 Alert Triage", "🌐 Global Explanations", "📊 Model Performance"]
    )

    # -----------------------------------------------------------------
    # TAB 1: Per-alert triage with SHAP + LIME
    # -----------------------------------------------------------------
    with tab_triage:
        st.subheader("Select an alert to triage")

        background = artifacts["background"]
        col_pick, col_detail = st.columns([1, 2])

        with col_pick:
            source = st.radio(
                "Alert source",
                ["Pick from sample alerts", "Upload / paste a flow"],
                label_visibility="collapsed",
            )

            if source == "Pick from sample alerts":
                idx = st.selectbox(
                    "Sample alert index",
                    options=list(range(len(background))),
                    format_func=lambda i: f"Alert #{i}",
                )
                raw_row = background.iloc[idx]
            else:
                uploaded = st.file_uploader("Upload a single-row CSV of flow features")
                if uploaded is not None:
                    raw_row = pd.read_csv(uploaded).iloc[0]
                else:
                    st.info("Upload a CSV, or switch back to sample alerts.")
                    st.stop()

        with col_detail:
            with st.expander("Raw flow features", expanded=False):
                st.dataframe(raw_row[meta["feature_names"]].to_frame("value"))

        pred_class, proba_dict, x_scaled_row = predict_alert(artifacts, raw_row)

        st.markdown("### Prediction")
        c1, c2 = st.columns([1, 2])
        with c1:
            severity = "🟢 Benign" if pred_class == "BENIGN" else "🔴 Malicious"
            st.metric("Predicted class", pred_class)
            st.write(severity)
        with c2:
            proba_series = pd.Series(proba_dict).sort_values(ascending=False).head(6)
            st.bar_chart(proba_series)

        st.markdown("---")
        st.markdown("### Why did the model predict this? (dual explanation)")

        if artifacts["shap_mode"] == "general":
            st.info(
                f"{meta.get('best_model_name')} isn't tree-based, so SHAP uses "
                "the slower general explainer here. This may take a little "
                "longer to render than a tree-based model would."
            )

        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            st.markdown("**SHAP (local, additive feature attribution)**")
            with st.spinner("Computing SHAP explanation..."):
                shap_row_exp = get_shap_explanation(artifacts, x_scaled_row, pred_class)
            fig = plt.figure()
            shap.plots.waterfall(shap_row_exp, show=False, max_display=10)
            st.pyplot(fig, clear_figure=True)

        with exp_col2:
            st.markdown("**LIME (local, surrogate-model attribution)**")
            with st.spinner("Computing LIME explanation..."):
                lime_exp = get_lime_explanation(artifacts, x_scaled_row, num_features=10)
            lime_df = pd.DataFrame(lime_exp.as_list(), columns=["condition", "weight"])
            lime_df = lime_df.sort_values("weight")
            fig2, ax = plt.subplots()
            colors = ["#d62728" if w < 0 else "#2ca02c" for w in lime_df["weight"]]
            ax.barh(lime_df["condition"], lime_df["weight"], color=colors)
            ax.set_xlabel("Contribution to predicted class")
            st.pyplot(fig2, clear_figure=True)

        st.caption(
            "SHAP and LIME are shown side-by-side (dual-explanation approach, "
            "MPR Section 3) so an analyst can cross-check whether both "
            "methods agree on the top drivers of this alert."
        )

    # -----------------------------------------------------------------
    # TAB 2: Global explanations
    # -----------------------------------------------------------------
    with tab_global:
        st.subheader("Global feature importance by attack class")
        class_choice = st.selectbox("Attack class", options=meta["classes"], index=0)
        max_n = 300 if artifacts["shap_mode"] == "tree" else 30
        n_bg = st.slider("Background sample size", 20, min(max_n, len(background)), min(100, max_n))

        with st.spinner("Computing global SHAP summary..."):
            shap_values, sample = get_global_shap_summary(
                artifacts, class_choice, sample_size=n_bg
            )
        fig3 = plt.figure()
        shap.summary_plot(shap_values, sample, show=False, max_display=15)
        st.pyplot(fig3, clear_figure=True)
        st.caption(
            f"SHAP summary plot for class '{class_choice}', computed on a "
            f"random sample of {len(sample)} background flows."
        )

    # -----------------------------------------------------------------
    # TAB 3: Model performance
    # -----------------------------------------------------------------
    with tab_model:
        st.subheader("Held-out test set performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        c2.metric("F1 (weighted)", f"{m['f1_weighted']:.3f}")
        c3.metric("F1 (macro)", f"{m['f1_macro']:.3f}")
        st.caption(
            "Macro F1 is typically much lower than weighted F1 on CICIDS "
            "due to severe class imbalance (rare attack classes like "
            "Heartbleed / SQL Injection / Infiltration have very few test "
            "instances)."
        )

        st.markdown("### All models compared")
        comparison = meta.get("all_models_comparison")
        if comparison:
            comp_df = pd.DataFrame(comparison).sort_values("f1_macro", ascending=False)
            st.dataframe(comp_df, width="stretch")
        else:
            st.info("No multi-model comparison table found in metadata.json.")

        st.markdown("**Model classes:**")
        st.write(", ".join(meta["classes"]))


if __name__ == "__main__":
    main()
