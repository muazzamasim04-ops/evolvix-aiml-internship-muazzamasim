import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="ML Model Inference App", page_icon="🤖", layout="centered")

@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except Exception:
    model = None

st.title("🤖 ML-Powered Web App UI")
st.write("Welcome! This interactive web interface wraps around our machine learning model to generate real-time predictions.")

st.subheader("Provide Input Features")
with st.form("prediction_form"):
    feature_1 = st.slider("Feature 1 (Numerical Parameter)", 0.0, 100.0, 25.0)
    feature_2 = st.selectbox("Feature 2 (Category)", ["Category A", "Category B", "Category C"])
    feature_3 = st.number_input("Feature 3 (Count/Value)", min_value=0, max_value=1000, value=50)
    
    submit_button = st.form_submit_button(label="Generate Prediction")

if submit_button:
    if model is None:
        st.error("Model file (`model.pkl`) not found! Please place your trained model in the directory.")
    else:
        cat_mapping = {"Category A": 0, "Category B": 1, "Category C": 2}
        input_data = np.array([[feature_1, cat_mapping[feature_2], feature_3]])
        
        try:
            prediction = model.predict(input_data)
            st.success("Prediction Generated Successfully!")
            st.metric(label="Predicted Output", value=str(prediction[0]))
        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
with st.expander("ℹ️ About this Model & Limitations"):
    st.markdown("""
    * **What the model does:** Analyzes structured input features to predict outcomes based on patterns learned during training.
    * **Key Features Used:** Primary numerical thresholds, categorical mappings, and scaled counts.
    * **Known Limitations & Edge Cases:** 
        * Performs poorly on out-of-distribution or extreme outlier data outside the training bounds.
        * Assumes independent and identically distributed (I.I.D.) inputs.
    """)
