# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# 2. Load the Dataset
try:
    df = pd.read_csv('Titanic-Dataset.csv')
except FileNotFoundError:
    print("Error: The 'Titanic-Dataset.csv' file was not found. Please make sure it's in the same directory as the script.")
    exit()

# --- Initial Data Exploration ---
print("--- Initial Data Info ---")
df.info()
print("\nMissing values before preprocessing:")
print(df.isnull().sum())
print("-" * 30)

# 3. Handle Missing Values
# Impute 'Age' with the median
imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])

# Impute 'Embarked' with the mode (most frequent value) and reassign
embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

# Drop the 'Cabin' column due to a large number of missing values
df.drop('Cabin', axis=1, inplace=True)

print("\n--- Data after Handling Missing Values ---")
df.info()
print("\nMissing values after preprocessing:")
print(df.isnull().sum())
print("-" * 30)

# 4. Visualize Outliers (for 'Fare') using a boxplot
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()

# 5. Encode Categorical Features
# 'Sex' and 'Embarked' are categorical. Use one-hot encoding.
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print("\n--- Data after One-Hot Encoding ---")
print(df.head())
print("-" * 30)

# 6. Feature Scaling
# Select numerical features to scale
numerical_features = ['Age', 'Fare']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\n--- Final Preprocessed Data (Scaled Features) ---")
print(df.head())
print("-" * 30)