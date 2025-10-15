import streamlit as st
import pandas as pd
from catboost import CatBoostClassifier, Pool
from io import StringIO
import os

# ===============================
# APP CONFIG
# ===============================
st.set_page_config(page_title="Company Status Prediction (No API Edition)", layout="wide")
st.title("Company Status Prediction (No API Edition)")
st.markdown("Predict the status of a company based on its financial data.")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    return model

model = load_model()
st.sidebar.success("✅ Model loaded successfully")

# Detect categorical features (optional)
model_features = model.feature_names_
cat_features = [f for f in model_features if "category" in f or "country" in f]
st.sidebar.write(f"🧠 Model expects {len(model_features)} features.")
st.sidebar.write("Categorical features:", cat_features)

# ===============================
# CHOOSE MODE
# ===============================
mode = st.radio("Choose Input Type", ["Manual Entry (Single Prediction)", "CSV Upload (Batch Predictions)"])

# ===============================
# 1️⃣ MANUAL ENTRY MODE
# ===============================
if mode == "Manual Entry (Single Prediction)":
    st.subheader("Enter Input Features")

    # Dynamic feature form based on model feature names
    input_data = {}
    for col in model_features:
        if col in cat_features:
            input_data[col] = st.text_input(f"{col} (categorical)", value="")
        else:
            input_data[col] = st.number_input(f"{col} (numeric)", value=0.0)

    # Predict button
    if st.button("🔮 Predict"):
        df = pd.DataFrame([input_data])

        # Handle missing closed_year or reorder columns
        if "closed_year" not in df.columns:
            df["closed_year"] = 0
        df = df.reindex(columns=model_features)

        pool = Pool(df, cat_features=cat_features)
        preds = model.predict(pool)
        st.success(f"✅ Prediction: {preds[0]}")
        st.dataframe(df)


# ===============================
# 2️⃣ CSV UPLOAD MODE
# ===============================
else:
    st.subheader("Upload CSV file for batch predictions")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Preview of your data:")
        st.dataframe(df.head())

        if st.button("📤 Predict CSV"):
            # Add dummy closed_year if missing
            if "closed_year" not in df.columns:
                df["closed_year"] = 0

            # Reorder columns
            df = df.reindex(columns=model_features)

            pool = Pool(df, cat_features=cat_features)
            preds = model.predict(pool)

            # ✅ Flatten predictions before assigning
            df["Prediction"] = preds.flatten()

            st.success("✅ Predictions completed!")
            st.dataframe(df)

            # Offer download button
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download Predictions CSV",
                csv,
                file_name="predictions.csv",
                mime="text/csv",
            )
