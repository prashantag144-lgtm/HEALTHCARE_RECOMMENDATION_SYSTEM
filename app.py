import streamlit as st
import pandas as pd
import joblib
import warnings

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Load Your Saved Assets (model.pkl, scaler.pkl, final_feature_names.pkl) ---
try:
    model = joblib.load('model.pkl')
    scaler = joblib.load('scaler.pkl')
    all_feature_names = joblib.load('final_feature_names.pkl')
    # This is the list of columns you scaled in your notebook
    columns_to_scale = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
except FileNotFoundError:
    st.error("Model assets not found! Please run your Jupyter Notebook to create the .pkl files first.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred loading assets: {e}")
    st.stop()


# --- 2. Your Recommendation Function (Copied from your notebook) ---
# This function is now the core of your web app
def generate_recommendations(patient_data, model, scaler, all_feature_names):
    
    patient_df = pd.DataFrame([patient_data])
    patient_processed = pd.get_dummies(patient_df)
    
    # Critical: Align columns to match the model's training data
    patient_aligned = patient_processed.reindex(columns=all_feature_names, fill_value=0)

    # Scale the required columns using the loaded scaler
    patient_aligned[columns_to_scale] = scaler.transform(patient_aligned[columns_to_scale])

    # Make prediction
    prediction = model.predict(patient_aligned)
    prediction_proba = model.predict_proba(patient_aligned)[0]

    # Return results
    return prediction[0], prediction_proba

# --- 3. Build the Streamlit User Interface ---

# Page configuration
st.set_page_config(page_title="Diabetes Predictor", layout="centered", initial_sidebar_state="auto")

# Title and introduction
st.title('🩺 Personalized Healthcare Recommendation System')
st.write("This application uses a Logistic Regression model (Accuracy: 95.7%) to predict a patient's risk of diabetes.")

# Sidebar for inputs (cleaner UI)
st.sidebar.header("Enter Patient Details:")

# Input fields in the sidebar
with st.sidebar:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    age = st.number_input("Age", min_value=0, max_value=120, value=45, help="Enter the patient's age in years.")
    bmi = st.number_input("BMI (Body Mass Index)", min_value=10.0, max_value=100.0, value=27.3, format="%.2f")
    
    st.markdown("---")
    
    HbA1c_level = st.number_input("HbA1c Level", min_value=3.0, max_value=15.0, value=6.0, format="%.1f")
    blood_glucose_level = st.number_input("Blood Glucose Level (mg/dL)", min_value=50, max_value=400, value=140)

    st.markdown("---")
    
    hypertension = st.selectbox("Hypertension", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
    heart_disease = st.selectbox("Heart Disease", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")

    st.markdown("---")

    smoking_history = st.selectbox("Smoking History", 
                                   ["never", "former", "current", "No Info", "not current", "ever"])


# Main page button
if st.sidebar.button("Get Recommendation", type="primary"):
    
    # Create the patient_data dictionary from UI inputs
    new_patient_data = {
        'gender': gender,
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'smoking_history': smoking_history,
        'bmi': bmi,
        'HbA1c_level': HbA1c_level,
        'blood_glucose_level': blood_glucose_level
    }

    # Get prediction
    prediction, prediction_proba = generate_recommendations(
        new_patient_data, model, scaler, all_feature_names
    )

    # Display the recommendation in the main area
    st.header("Prediction Result:")
    
    if prediction == 0:
        confidence = prediction_proba[0] * 100
        st.success(f"**Prediction: No Diabetes (Low Risk)**")
        st.metric(label="Confidence", value=f"{confidence:.1f}%")
        st.write("Recommendation: Continue to maintain a healthy lifestyle, diet, and regular exercise.")
    else:
        confidence = prediction_proba[1] * 100
        st.error(f"**Prediction: Diabetes (High Risk)**")
        st.metric(label="Confidence", value=f"{confidence:.1f}%")
        st.warning("**Recommendation: CRITICAL! Please schedule an appointment with a healthcare professional for a formal diagnosis and treatment plan.**")

else:
    st.info("Please enter the patient's details in the sidebar and click 'Get Recommendation'.")