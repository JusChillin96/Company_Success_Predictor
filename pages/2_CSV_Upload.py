import streamlit as st
import pandas as pd
from catboost import Pool
from Homepage import model  # reuse cached model from main app

# -------------------------------
# PAGE SETUP
# -------------------------------
st.title("📁 CSV Upload Prediction")

# Get model info
model_features = model.feature_names_
cat_features = [f for f in model_features if "category" in f or "country" in f]

st.markdown("Upload a CSV file containing the input features for batch prediction.")

# -------------------------------
# FILE UPLOAD
# -------------------------------
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Preview of Uploaded Data:")
    st.dataframe(df)

    # -------------------------------
    # PREDICTION
    # -------------------------------
    if st.button("📤 Predict"):
        # Ensure model expects all features
        if "closed_year" not in df.columns:
            df["closed_year"] = 0  # Dummy column if missing

        # Reorder to match training feature order
        df = df.reindex(columns=model_features)

        # Create CatBoost Pool
        pool = Pool(df, cat_features=cat_features)

        # Get predictions
        preds = model.predict(pool)

        # Flatten to 1D (avoids ValueError)
        df["Prediction"] = preds.ravel()

        # Show results
        st.success("✅ Predictions completed successfully!")
        st.dataframe(df)

        # -------------------------------
        # DOWNLOAD RESULTS
        # -------------------------------
        csv = df.to_csv(index=False)
        st.download_button(
            label="⬇️ Download Predictions CSV",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
        )

else:
    st.info("👆 Upload a CSV file above to begin predictions.")