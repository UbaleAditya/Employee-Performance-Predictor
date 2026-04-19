import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Create outputs folder for GitHub proof [cite: 93]
if not os.path.exists('outputs'):
    os.makedirs('outputs')

# Load the data [cite: 260]
df = pd.read_csv('data/employee_data.csv')
sns.set_theme(style="whitegrid")

# Plot 1: Performance Bands Distribution [cite: 281]
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x='Performance_Band', palette='viridis', hue='Performance_Band', legend=False)
plt.title('Performance Band Distribution')
plt.savefig('outputs/performance_distribution.png')

# Plot 2: Manager Score vs Performance [cite: 284]
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='Performance_Band', y='Manager_Score', palette='Set2', hue='Performance_Band', legend=False)
plt.title('How Manager Scores Influence Performance Bands')
plt.savefig('outputs/manager_impact.png')

print("✅ Success: Graphs saved in 'outputs/' folder!")