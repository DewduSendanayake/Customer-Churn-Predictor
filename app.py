import streamlit as st
import pickle
from pathlib import Path
import pandas as pd

# Paths
ROOT          = Path.cwd().parent
MODEL_PATH    = ROOT/'models'/'xgb.pkl'
SCALER_PATH   = ROOT/'models'/'scaler.pkl'
FEATURES_PATH = ROOT/'models'/'feature_names.pkl'

# Load artifacts
model         = pickle.load(open(MODEL_PATH,    'rb'))
scaler        = pickle.load(open(SCALER_PATH,   'rb'))
feature_names = pickle.load(open(FEATURES_PATH, 'rb'))

# 1) Gather raw inputs
# st.title("Customer Churn Predictor 💔")

st.markdown(
    "<h1 style='text-align: center; color: #fcefee;'>Customer Churn Predictor 🪄</h1>",
    unsafe_allow_html=True
)


tenure = st.slider("Tenure (months)", 0, 72, 12)
monthly = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total = st.number_input("Total Charges", 0.0, 10000.0, 1500.0)
partner = st.selectbox("Has Partner?", ["No","Yes"])
dependents = st.selectbox("Has Dependents?", ["No","Yes"])
# …add any other categorical widgets exactly once…

# 2) Put them into a dict, using the *original* column names* where possible
raw = {
  'tenure': tenure,
  'MonthlyCharges': monthly,
  'TotalCharges': total,
  'Partner': 1 if partner=="Yes" else 0,
  'Dependents': 1 if dependents=="Yes" else 0,
  # …for any other label‑encoded features, map similarly…
}

# 3) Build a one‑row DataFrame and then one‑hot encode any remaining categoricals:
input_df = pd.DataFrame([raw])
# If you used pd.get_dummies() during training, apply it here too:
input_df = pd.get_dummies(input_df)

# 4) Reindex to the saved feature list, filling missing cols with 0
input_aligned = input_df.reindex(columns=feature_names, fill_value=0)

# 5) Scale numeric columns
num_cols = ['tenure','MonthlyCharges','TotalCharges']
input_aligned[num_cols] = scaler.transform(input_aligned[num_cols])

# 6) Finally predict
if st.button("Predict Churn Risk"):
    prob = model.predict_proba(input_aligned)[:,1][0]
    st.metric("Churn Risk Score", f"{prob:.2%}")
