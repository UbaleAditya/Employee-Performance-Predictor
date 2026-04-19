import streamlit as st
import pandas as pd
import joblib

# Load the model
model = joblib.load('models/performance_model.pkl')

st.title("🚀 Employee Performance Predictor")
st.write("Enter employee details below to predict their performance band.")

# Create input form [cite: 411]
with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30)
        gender = st.selectbox("Gender", ["Male", "Female"])
        dept = st.selectbox("Department", ["IT", "HR", "Sales", "Finance", "Marketing"])
        exp = st.number_input("Years of Experience", 0, 40, 5)
    
    with col2:
        projects = st.slider("Projects Count", 1, 15, 5)
        training = st.slider("Training Hours", 10, 100, 50)
        delay = st.slider("Avg Task Delay (Days)", 0, 20, 2)
        m_score = st.slider("Manager Score", 1.0, 5.0, 3.5)
        p_feedback = st.slider("Peer Feedback", 1.0, 5.0, 3.5)

    submit = st.form_submit_button("Predict Performance")
if submit:
    # Prepare data for prediction
    input_data = pd.DataFrame({
        'Age': [age], 'Gender': [gender], 'Department': [dept],
        'Experience_Years': [exp], 'Projects_Count': [projects],
        'Training_Hours': [training], 'Avg_Task_Delay': [delay],
        'Manager_Score': [m_score], 'Peer_Feedback': [p_feedback]
    })
    
    # Get Prediction [cite: 413-414]
    res = model.predict(input_data)[0]
    
    # Display Result
    st.success(f"The Predicted Performance Band is: **{res}**")