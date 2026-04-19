import pandas as pd
import numpy as np
import os

# 1. Create the data folder if it doesn't exist
if not os.path.exists('data'):
    os.makedirs('data')

# 2. Set seed for reproducibility (so you get the same 'random' data every time)
np.random.seed(42)

# 3. Configuration
num_employees = 1000

# 4. Generating Synthetic Features [cite: 239-248]
data = {
    'Employee_ID': range(1001, 1001 + num_employees),
    'Age': np.random.randint(22, 60, size=num_employees),
    'Gender': np.random.choice(['Male', 'Female'], size=num_employees),
    'Department': np.random.choice(['IT', 'HR', 'Sales', 'Finance', 'Marketing'], size=num_employees),
    'Experience_Years': np.random.randint(1, 20, size=num_employees),
    'Projects_Count': np.random.randint(1, 10, size=num_employees),
    'Training_Hours': np.random.randint(10, 100, size=num_employees),
    'Avg_Task_Delay': np.random.randint(0, 15, size=num_employees),
    'Manager_Score': np.random.uniform(1, 5, size=num_employees).round(1),
    'Peer_Feedback': np.random.uniform(1, 5, size=num_employees).round(1),
}

df = pd.DataFrame(data)

# 5. Logic for the Prediction Target (Performance_Band) [cite: 227, 249]
# We calculate a hidden 'score' to make the data learnable by the AI
score = (df['Manager_Score'] * 0.4) + (df['Peer_Feedback'] * 0.3) + \
        (df['Projects_Count'] * 0.2) - (df['Avg_Task_Delay'] * 0.1)

# Categorize into High, Medium, Low bands [cite: 237]
df['Performance_Band'] = pd.qcut(score, q=3, labels=['Low', 'Medium', 'High'])

# 6. Save to the data folder
df.to_csv('data/employee_data.csv', index=False)
print("✅ Success: Synthetic dataset created at 'data/employee_data.csv'!")