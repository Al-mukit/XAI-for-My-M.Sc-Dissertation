# XAI-SOC Streamlit Dashboard

Analyst-facing frontend + backend for alert triage with SHAP + LIME
explanations, reading the artifacts your notebook already exported.

## Setup

1. Put your `artifacts/` folder (the one with `best_model.joblib`,
   `scaler.joblib`, `label_encoder.joblib`, `background_sample.csv`,
   `metadata.json`) directly inside this `streamlit_app/` folder, next to
   `app.py`. (You said you already have this — just drop it in here, this
   zip does not include or overwrite it.)

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run:
   ```bash
   streamlit run app.py
   ```
   Opens at `http://localhost:8501`.

If your artifacts live somewhere else, set an environment variable instead
of moving them:
```bash
export XAI_SOC_ARTIFACT_DIR=/path/to/your/artifacts   # Windows: set XAI_SOC_ARTIFACT_DIR=...
streamlit run app.py
```

## What's inside

- `app.py` — the dashboard UI, 3 tabs:
  - **Alert Triage** — pick a sample alert (or upload one), see the
    prediction, and a side-by-side SHAP waterfall + LIME bar chart.
  - **Global Explanations** — SHAP summary plot per attack class.
  - **Model Performance** — test metrics + the full model-comparison table
    (from `metadata.json`'s `all_models_comparison`, if your notebook run
    produced one).
- `modules/backend.py` — all non-UI logic: loading artifacts, generating
  predictions, SHAP, and LIME. Automatically detects whether your winning
  model was tree-based (uses SHAP's fast `TreeExplainer`) or not
  (Logistic Regression / KNN / MLP — uses the slower general `Explainer`
  with smaller sample sizes, same tradeoff discussed in your MPR).
  Also handles both `.joblib` (sklearn/XGBoost) and `.keras` (MLP) model
  files transparently, based on `metadata.json`'s `best_model_file` field.

## Notes

- If your best model was the MLP, make sure `tensorflow` is installed
  (it's in `requirements.txt`) — the app loads `.keras` files with it.
- The "Global Explanations" tab caps the background sample size much lower
  for non-tree models (30 vs 300), since the general SHAP explainer is
  expensive per call — this mirrors the slowness issue you hit earlier
  with KNN.
