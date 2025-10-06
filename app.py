# app.py
import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
import base64

from io import StringIO

# --- Load trained model and preprocessing objects ---
model = joblib.load("xgboost_multiclass_model_ensemble.pkl")
imputer = joblib.load("imputer.pkl")
label_enc = joblib.load("label_enc.pkl")  # ✅ FIXED filename

# --- SHAP Explainer (TreeExplainer for XGBoost) ---
explainer = shap.TreeExplainer(model)

# --- Features used by model ---
features = ['koi_period', 'koi_prad', 'koi_duration', 'koi_depth', 'koi_ror',
            'koi_sma', 'koi_insol', 'koi_teq', 'koi_incl', 'koi_model_snr',
            'koi_steff', 'koi_slogg', 'koi_smet', 'koi_srad', 'koi_smass', 'period_prad']

# --- Default input values ---
default_input = {
    'koi_period': 12.5,
    'koi_prad': 1.9,
    'koi_duration': 5.4,
    'koi_depth': 0.0012,
    'koi_ror': 0.017,
    'koi_sma': 0.1,
    'koi_insol': 180.0,
    'koi_teq': 600,
    'koi_incl': 89.5,
    'koi_model_snr': 15.0,
    'koi_steff': 5500,
    'koi_slogg': 4.4,
    'koi_smet': 0.0,
    'koi_srad': 1.0,
    'koi_smass': 1.0
}

# --- Streamlit Page Config ---
st.set_page_config(page_title="Exoplanet Classifier", layout="wide")
st.title("🪐 Exoplanet Classification Web App")

st.markdown("""
This tool uses a machine learning model trained on NASA's **Kepler** mission data to predict whether an observed celestial object is a:
- ✅ Confirmed Exoplanet
- ❓ Candidate
- ❌ False Positive
""")

# --- Sidebar for input ---
st.sidebar.header("Manual Feature Input")

input_data = {}
for feature in features[:-1]:  # Exclude interaction term
    input_data[feature] = st.sidebar.number_input(
        f"{feature}",
        value=default_input.get(feature, 0.0)
    )

# --- Calculate interaction term ---
input_data['period_prad'] = input_data['koi_period'] * input_data['koi_prad']

# --- Predict button ---
if st.sidebar.button("Predict from Input"):
    input_df = pd.DataFrame([input_data])
    input_array = imputer.transform(input_df[features])

    # Predict
    pred_idx = model.predict(input_array)[0]
    pred_proba = model.predict_proba(input_array)[0]
    pred_class = label_enc.inverse_transform([pred_idx])[0]

    st.subheader("Prediction Result:")
    st.write(f"**Predicted Class:** `{pred_class}`")

    st.subheader("Class Probabilities:")
    prob_df = pd.DataFrame({'Class': label_enc.classes_, 'Probability': pred_proba})
    st.bar_chart(prob_df.set_index("Class"))

    # --- SHAP Explainability ---
    st.subheader("Model Explainability (SHAP)")
    st.write("Feature contribution for the predicted class:")

    # Handle multiclass SHAP output
    shap_values = explainer.shap_values(input_array)

    # Show only SHAP values for the predicted class
    predicted_shap_values = shap_values[pred_idx]

    #fig, ax = plt.subplots()
    #shap.summary_plot(predicted_shap_values, input_array, feature_names=features, plot_type="bar", show=False)
    #st.pyplot(fig)
    
    shap_values_single = explainer(input_array)

    # SHAP bar plot (for single prediction explanation)
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.bar(shap_values_single[0], show=False)
    st.pyplot(bbox_inches='tight')


# --- CSV Batch Upload ---
st.header("Batch Predictions via CSV Upload")
st.markdown("Upload a CSV file with the same columns used by the model:")

csv_file = st.file_uploader("Upload your CSV", type=['csv'])

if csv_file is not None:
    try:
        df = pd.read_csv(csv_file)

        if not all(f in df.columns for f in features[:-1]):
            st.error("CSV missing required columns. Please include all features used by the model.")
        else:
            df['period_prad'] = df['koi_period'] * df['koi_prad']
            X = imputer.transform(df[features])
            preds = model.predict(X)
            probs = model.predict_proba(X)
            pred_labels = label_enc.inverse_transform(preds)

            result_df = df.copy()
            result_df['Predicted Class'] = pred_labels

            for i, cls in enumerate(label_enc.classes_):
                result_df[f"Prob_{cls}"] = probs[:, i]

            st.success("Predictions completed!")
            st.dataframe(result_df.head())

            # Downloadable CSV
            csv = result_df.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}" download="predictions.csv">📥 Download Predictions CSV</a>'
            st.markdown(href, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error processing file: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("Built using data from [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).")
