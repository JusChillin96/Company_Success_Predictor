import streamlit as st
from catboost import CatBoostClassifier

st.set_page_config(page_title="Company Success Prediction App", layout="wide")

st.title("Homepage")
st.markdown("""
Welcome to the **Company Success Prediction App**.

Use the sidebar to choose between:
- 🧍 Manual single prediction  
- 📁 CSV batch prediction
""")

@st.cache_resource
def load_model():
    model = CatBoostClassifier()
    model.load_model("catboost_model.cbm")
    return model

# Cache the model globally so both pages can reuse it
model = load_model()
st.sidebar.success("✅ Model loaded successfully")
