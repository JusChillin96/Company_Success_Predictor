import streamlit as st
import pandas as pd
from catboost import Pool
from Homepage import model  # reuse the cached model

# -------------------------------
# PAGE SETUP
# -------------------------------
st.title("🧍 Manual Entry Prediction")

# Model info
model_features = model.feature_names_
cat_features = [f for f in model_features if "category" in f or "country" in f]

# -------------------------------
# FEATURE DESCRIPTIONS
# -------------------------------
feature_descriptions = {
    "active_days": "Total number of days the company has been active since founding.",
    "closed_year": "The year the company closed operations (0 if still active).",
    "founded_at": "Year the company was founded.",
    "category_code": "Main industry or business category (e.g., software, biotech, web).",
    "relationships": "Number of business relationships or partnerships associated with the company.",
    "country_code": "Country code where the company is based (e.g., US, GB, IN).",
    "last_funding_at": "Date (or year) when the company last received funding.",
    "milestones": "Total number of milestones or major achievements reached by the company.",
    "funding_total_usd": "Total funding amount received by the company (in USD).",
    "lat": "Latitude coordinate of the company's primary office location.",
    "first_funding_at": "Date (or year) when the company first received funding.",
    "first_milestone_at": "Date (or year) of the company's first recorded milestone.",
    "lng": "Longitude coordinate of the company's primary office location.",
    "last_milestone_at": "Date (or year) of the company's most recent milestone.",
    "funding_rounds": "Total number of funding rounds participated in by the company.",
    "investment_rounds": "Total number of investment rounds involving the company.",
}

# -------------------------------
# INPUT FORM
# -------------------------------
st.markdown("Fill in the following details for a single prediction:")

input_data = {}
for col in model_features:
    desc = feature_descriptions.get(col, "")
    if col in cat_features:
        input_data[col] = st.text_input(f"{col} (categorical)", value="")
    else:
        input_data[col] = st.number_input(f"{col} (numeric)", value=0.0)
    if desc:
        st.caption(f"🛈 {desc}")

# -------------------------------
# PREDICT BUTTON
# -------------------------------
if st.button("🔮 Predict"):
    df = pd.DataFrame([input_data])

    if "closed_year" not in df.columns:
        df["closed_year"] = 0

    df = df.reindex(columns=model_features)

    pool = Pool(df, cat_features=cat_features)
    preds = model.predict(pool)

    st.success(f"✅ Prediction: {preds[0]}")
    st.dataframe(df)
