import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt

# --- Load your dataset as before ---
# Assume 'df' is loaded DataFrame with features and label column 'koi_pdisposition'


df = pd.read_csv('kepler.csv')
# Select relevant features - add some interaction terms for feature engineering
features = ['koi_period', 'koi_prad', 'koi_duration', 'koi_depth', 'koi_ror', 
            'koi_sma', 'koi_insol', 'koi_teq', 'koi_incl', 'koi_model_snr',
            'koi_steff', 'koi_slogg', 'koi_smet', 'koi_srad', 'koi_smass']

# Add simple interaction feature example: period * radius
df['period_prad'] = df['koi_period'] * df['koi_prad']
features.append('period_prad')

# Prepare data
X = df[features]
y = df['koi_pdisposition']

# Encode labels to numeric (e.g. 0,1,2)
label_enc = LabelEncoder()
y_encoded = label_enc.fit_transform(y)

# Handle missing data
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Handle imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# --- Ensemble Models ---
models = {
    'RandomForest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'GradientBoosting': GradientBoostingClassifier(random_state=42),
    'XGBoost': xgb.XGBClassifier(eval_metric='mlogloss', random_state=42)
}

# Hyperparameter grids for tuning (basic example)
param_grids = {
    'RandomForest': {'n_estimators': [100, 200], 'max_depth': [10, 20]},
    'AdaBoost': {'n_estimators': [50, 100], 'learning_rate': [0.5, 1.0]},
    'GradientBoosting': {'n_estimators': [100, 150], 'learning_rate': [0.1, 0.05]},
    'XGBoost': {'n_estimators': [100, 150], 'learning_rate': [0.1, 0.05], 'max_depth': [3, 5]}
}

best_models = {}
for name, model in models.items():
    print(f"Training and tuning {name}...")
    grid = GridSearchCV(model, param_grids[name], cv=3, scoring='accuracy', n_jobs=-1)
    grid.fit(X_train_res, y_train_res)
    best_models[name] = grid.best_estimator_
    print(f"Best params for {name}: {grid.best_params_}")

# Evaluate models on test set
for name, model in best_models.items():
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n{name} Accuracy: {acc:.4f}")
    print(f"Classification Report for {name}:\n{classification_report(y_test, y_pred, target_names=label_enc.classes_)}")
    print(f"Confusion Matrix for {name}:\n{confusion_matrix(y_test, y_pred)}")

# --- SHAP Explainability for the best XGBoost model ---
best_xgb = best_models['XGBoost']

#explainer = shap.Explainer(best_xgb)
#shap_values = explainer(X_test)

# Summary plot
#shap.summary_plot(shap_values, features=X_test, feature_names=features)

# --- Save the best XGBoost model for inference ---
import joblib
joblib.dump(best_xgb, 'xgboost_multiclass_model_ensemble.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(label_enc, 'label_enc.pkl')


# --- Predict function example ---
def predict_planet(input_features_dict):
    # Prepare input array
    input_df = pd.DataFrame([input_features_dict])
    input_df['period_prad'] = input_df['koi_period'] * input_df['koi_prad']
    input_array = imputer.transform(input_df[features])
    pred_class_idx = best_xgb.predict(input_array)[0]
    pred_proba = best_xgb.predict_proba(input_array)[0]

    pred_class = label_enc.inverse_transform([pred_class_idx])[0]
    return pred_class, pred_proba

# Example input for testing (based on your sample data, fill with real values)

sample_input = {
    'koi_period': 3.0,
    'koi_prad': 1.49,
    'koi_duration': 19.9,
    'koi_depth': 0.0341,
    'koi_ror': 0.154,
    'koi_sma': 0.969,
    'koi_insol': 5.03,
    'koi_teq': 638,
    'koi_incl': 88.96,
    'koi_model_snr': 14.6,
    'koi_steff': 297.0,
    'koi_slogg': 4.54,
    'koi_smet': 0.042,
    'koi_srad': 0.271,
    'koi_smass': 0.3858
}
predicted_class, predicted_probabilities = predict_planet(sample_input)
print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {predicted_probabilities}")



#candidates ammple input
sample_input = {
    'koi_period': 12.5,        # Medium orbital period
    'koi_prad': 1.9,           # Reasonable planet size
    'koi_duration': 5.4,       # Duration of transit
    'koi_depth': 0.0012,       # Shallow-ish transit (not too deep)
    'koi_ror': 0.017,          # Radius ratio (planet/star)
    'koi_sma': 0.1,            # Semi-major axis
    'koi_insol': 180.0,        # Stellar insolation
    'koi_teq': 600,            # Equilibrium temperature
    'koi_incl': 89.5,          # Close to edge-on orbit
    'koi_model_snr': 15.0,     # Signal-to-noise (not very high)
    'koi_steff': 5500,         # Effective temperature of star
    'koi_slogg': 4.4,          # Log(g)
    'koi_smet': 0.0,           # Metallicity (solar)
    'koi_srad': 1.0,           # Star radius in solar radii
    'koi_smass': 1.0           # Star mass in solar masses
}


predicted_class, predicted_probabilities = predict_planet(sample_input)
print(f"Predicted class: {predicted_class}")
print(f"Class probabilities: {predicted_probabilities}")

