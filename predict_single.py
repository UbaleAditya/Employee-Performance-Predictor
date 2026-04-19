import pandas as pd
import joblib

# 1. Load the trained model pipeline [cite: 409]
model = joblib.load('models/performance_model.pkl')

# 2. Simulate a "New Employee" data entry [cite: 411]
new_employee = pd.DataFrame({
    'Age': [30],
    'Gender': ['Male'],
    'Department': ['IT'],
    'Experience_Years': [5],
    'Projects_Count': [8],
    'Training_Hours': [80],
    'Avg_Task_Delay': [1],
    'Manager_Score': [4.5],
    'Peer_Feedback': [4.2]
})

# 3. Make Prediction [cite: 413]
prediction = model.predict(new_employee)

print("--- HR Prediction Result ---")
print(f"Predicted Performance Band: {prediction[0]}") # [cite: 414]