import pandas as pd

# Load the dataset [cite: 260]
df = pd.read_csv('data/employee_data.csv')

# 1. Show the first 5 rows [cite: 185]
print("--- Dataset Preview ---")
print(df.head())

# 2. Check for missing values [cite: 256]
print("\n--- Missing Values ---")
print(df.isnull().sum())

# 3. Check the distribution of our Target [cite: 278]
print("\n--- Performance Band Counts ---")
print(df['Performance_Band'].value_counts())