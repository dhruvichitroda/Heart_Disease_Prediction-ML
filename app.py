"""
Heart Disease Prediction - Streamlit Web Application
=====================================================

This Streamlit app allows users to input patient data and predict
the risk of heart disease using a trained machine learning model.

Author: Generated for Heart Disease Prediction Project
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        padding: 2rem;
        border-radius: 10px;
        margin: 2rem 0;
        text-align: center;
    }
    .high-risk {
        background-color: #FFE5E5;
        border: 3px solid #FF6B6B;
    }
    .low-risk {
        background-color: #E5F5F3;
        border: 3px solid #4ECDC4;
    }
    .metric-card {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 5px;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================
@st.cache_resource
def load_model():
    """Load the trained model from disk"""
    try:
        model_path = 'models/heart_disease_model.pkl'
        if not os.path.exists(model_path):
            st.error(f"❌ Model file not found at {model_path}")
            st.info("Please run 'train_model.py' first to train and save the model.")
            return None, None
        
        model = joblib.load(model_path)
        
        # Load metadata
        metadata_path = 'models/model_metadata.json'
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = None
            
        return model, metadata
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

# Load model
model, metadata = load_model()

# ============================================================================
# MAIN HEADER
# ============================================================================
st.markdown('<h1 class="main-header">❤️ Heart Disease Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Enter patient information below to predict the risk of heart disease</p>', unsafe_allow_html=True)

# ============================================================================
# SIDEBAR - MODEL INFORMATION
# ============================================================================
with st.sidebar:
    st.header("📊 Model Information")
    
    if metadata:
        st.success(f"✅ Model: {metadata.get('model_name', 'Unknown')}")
        st.metric("Accuracy", f"{metadata.get('accuracy', 0)*100:.2f}%")
        st.metric("F1-Score", f"{metadata.get('f1_score', 0):.4f}")
        st.metric("Precision", f"{metadata.get('precision', 0):.4f}")
        st.metric("Recall", f"{metadata.get('recall', 0):.4f}")
    else:
        st.warning("Model metadata not available")
    
    st.markdown("---")
    st.header("📋 Feature Descriptions")
    st.markdown("""
    **age**: Age in years  
    **sex**: Gender (1 = Male, 0 = Female)  
    **cp**: Chest pain type (0-3)  
    **trestbps**: Resting blood pressure (mm Hg)  
    **chol**: Serum cholesterol (mg/dl)  
    **fbs**: Fasting blood sugar > 120 mg/dl (1 = Yes, 0 = No)  
    **restecg**: Resting ECG results (0-2)  
    **thalach**: Maximum heart rate achieved  
    **exang**: Exercise induced angina (1 = Yes, 0 = No)  
    **oldpeak**: ST depression induced by exercise  
    **slope**: Slope of peak exercise ST segment (0-2)  
    **ca**: Number of major vessels colored by flourosopy (0-3)  
    **thal**: Thalassemia (0-3)
    """)

# ============================================================================
# MAIN CONTENT - INPUT FORM
# ============================================================================
if model is None:
    st.stop()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Patient Information")
    
    # Age
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=50,
        help="Patient's age in years"
    )
    
    # Sex
    sex = st.selectbox(
        "Sex",
        options=[0, 1],
        format_func=lambda x: "Female" if x == 0 else "Male",
        help="Patient's gender"
    )
    
    # Chest Pain Type (cp)
    cp = st.selectbox(
        "Chest Pain Type (cp)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Typical Angina",
            1: "Atypical Angina",
            2: "Non-anginal Pain",
            3: "Asymptomatic"
        }.get(x, x),
        help="Type of chest pain experienced"
    )
    
    # Resting Blood Pressure
    trestbps = st.number_input(
        "Resting Blood Pressure (mm Hg)",
        min_value=50,
        max_value=250,
        value=120,
        help="Resting blood pressure in millimeters of mercury"
    )
    
    # Serum Cholesterol
    chol = st.number_input(
        "Serum Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=200,
        help="Serum cholesterol level in mg/dl"
    )
    
    # Fasting Blood Sugar
    fbs = st.selectbox(
        "Fasting Blood Sugar > 120 mg/dl",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Is fasting blood sugar > 120 mg/dl?"
    )
    
    # Resting ECG
    restecg = st.selectbox(
        "Resting ECG Results",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Normal",
            1: "ST-T Wave Abnormality",
            2: "Left Ventricular Hypertrophy"
        }.get(x, x),
        help="Resting electrocardiographic results"
    )

with col2:
    st.subheader("🏥 Medical Test Results")
    
    # Maximum Heart Rate
    thalach = st.number_input(
        "Maximum Heart Rate Achieved",
        min_value=60,
        max_value=220,
        value=150,
        help="Maximum heart rate achieved during exercise"
    )
    
    # Exercise Induced Angina
    exang = st.selectbox(
        "Exercise Induced Angina",
        options=[0, 1],
        format_func=lambda x: "No" if x == 0 else "Yes",
        help="Does exercise cause chest pain?"
    )
    
    # ST Depression
    oldpeak = st.number_input(
        "ST Depression (oldpeak)",
        min_value=0.0,
        max_value=10.0,
        value=1.0,
        step=0.1,
        format="%.1f",
        help="ST depression induced by exercise relative to rest"
    )
    
    # Slope
    slope = st.selectbox(
        "Slope of Peak Exercise ST Segment",
        options=[0, 1, 2],
        format_func=lambda x: {
            0: "Upsloping",
            1: "Flat",
            2: "Downsloping"
        }.get(x, x),
        help="Slope of the peak exercise ST segment"
    )
    
    # Number of Major Vessels
    ca = st.selectbox(
        "Number of Major Vessels (ca)",
        options=[0, 1, 2, 3],
        help="Number of major vessels colored by flourosopy"
    )
    
    # Thalassemia
    thal = st.selectbox(
        "Thalassemia (thal)",
        options=[0, 1, 2, 3],
        format_func=lambda x: {
            0: "Normal",
            1: "Fixed Defect",
            2: "Reversible Defect",
            3: "Unknown"
        }.get(x, x),
        help="Thalassemia type"
    )

# ============================================================================
# PREDICTION BUTTON AND RESULTS
# ============================================================================
st.markdown("---")

# Center the predict button
col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
with col_btn2:
    predict_button = st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True)

if predict_button:
    # Create input DataFrame with exact column order as training data
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'cp': [cp],
        'trestbps': [trestbps],
        'chol': [chol],
        'fbs': [fbs],
        'restecg': [restecg],
        'thalach': [thalach],
        'exang': [exang],
        'oldpeak': [oldpeak],
        'slope': [slope],
        'ca': [ca],
        'thal': [thal]
    })
    
    # Ensure column order matches training data
    if metadata and 'feature_names' in metadata:
        input_data = input_data[metadata['feature_names']]
    
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        
        # Risk level and probability
        risk_level = "High Risk" if prediction == 1 else "Low Risk"
        risk_probability = prediction_proba[1] if prediction == 1 else prediction_proba[0]
        
        # Styled prediction box
        if prediction == 1:
            st.markdown(
                f"""
                <div class="prediction-box high-risk">
                    <h2 style="color: #FF6B6B; font-size: 2.5rem;">⚠️ HIGH RISK OF HEART DISEASE</h2>
                    <p style="font-size: 1.5rem; margin-top: 1rem; color: #2c3e50; font-weight: 600;">
                        Risk Probability: <strong style="color: #c0392b; font-size: 1.8rem;">{risk_probability*100:.2f}%</strong>
                    </p>
                    <p style="margin-top: 1rem; color: #666;">
                        Please consult with a healthcare professional for further evaluation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="prediction-box low-risk">
                    <h2 style="color: #4ECDC4; font-size: 2.5rem;">✅ LOW RISK OF HEART DISEASE</h2>
                    <p style="font-size: 1.5rem; margin-top: 1rem; color: #2c3e50; font-weight: 600;">
                        Risk Probability: <strong style="color: #27ae60; font-size: 1.8rem;">{risk_probability*100:.2f}%</strong>
                    </p>
                    <p style="margin-top: 1rem; color: #666;">
                        Continue maintaining a healthy lifestyle!
                    </p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        # Show probability breakdown
        st.markdown("### 📊 Prediction Probabilities")
        prob_col1, prob_col2 = st.columns(2)
        
        with prob_col1:
            st.metric(
                "Low Risk Probability",
                f"{prediction_proba[0]*100:.2f}%",
                delta=None
            )
        
        with prob_col2:
            st.metric(
                "High Risk Probability",
                f"{prediction_proba[1]*100:.2f}%",
                delta=None
            )
        
        # Show input summary
        with st.expander("📋 View Input Summary"):
            st.dataframe(input_data.T, use_container_width=True)
            
    except Exception as e:
        st.error(f"❌ Error making prediction: {str(e)}")
        st.info("Please check that all input fields are filled correctly.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p><strong>⚠️ Disclaimer:</strong> This prediction is for educational purposes only.</p>
        <p>Always consult with qualified healthcare professionals for medical advice.</p>
        <p style="margin-top: 1rem;">Heart Disease Prediction System © 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)
