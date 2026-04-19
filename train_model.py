import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# Ensure models folder exists [cite: 92]
if not os.path.exists('models'):
    os.makedirs('models')

# Load data [cite: 260]
df = pd.read_csv('data/employee_data.csv')

# Define Features and Target [cite: 306, 308]
X = df.drop(['Employee_ID', 'Performance_Band'], axis=1)
y = df['Performance_Band']

# Preprocessing Pipeline [cite: 298, 299, 311-318]
cat_cols = ['Gender', 'Department']
num_cols = ['Age', 'Experience_Years', 'Projects_Count', 'Training_Hours', 'Avg_Task_Delay', 'Manager_Score', 'Peer_Feedback']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), num_cols),
    ('cat', OneHotEncoder(), cat_cols)
])

# Define and Train Model [cite: 336, 444]
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
model.fit(X_train, y_train)

# Save the model for GitHub [cite: 448]
joblib.dump(model, 'models/performance_model.pkl')
print("✅ Model saved to 'models/performance_model.pkl'")

# Final Report [cite: 359]
print("\n--- Model Performance Report ---")
print(classification_report(y_test, model.predict(X_test)))