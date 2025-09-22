# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 2. Load the Dataset
# Make sure the 'titanic_dataset.csv' is in the same directory as your script
try:
    df = pd.read_csv('titanic_dataset.csv')
except FileNotFoundError:
    print("Dataset not found. Please download the Titanic dataset and place it in the same folder.")
    exit()

# 3. Initial Exploration
print("Initial Data Info:")
df.info()
print("\nMissing Values:")
print(df.isnull().sum())
print("\n")

# 4. Handle Missing Values
# Impute 'Age' with the median
imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])

# Impute 'Embarked' with the mode (most frequent value)
imputer_embarked = SimpleImputer(strategy='most_frequent')
df['Embarked'] = imputer_embarked.fit_transform(df[['Embarked']])

# Drop the 'Cabin' column due to a large number of missing values
df.drop('Cabin', axis=1, inplace=True)

print("Data Info after Handling Missing Values:")
df.info()
print("\n")

# 5. Handle Outliers in 'Fare' using a boxplot for visualization
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()

# You can choose to remove outliers, but for this exercise, we will just visualize them.
# A common method is the IQR (Interquartile Range) method, but for simplicity, we'll proceed.

# 6. Encode Categorical Features
# Use one-hot encoding for 'Sex' and 'Embarked'
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Note: 'drop_first=True' avoids the dummy variable trap.

print("Data Info after One-Hot Encoding:")
df.info()
print("\n")

# 7. Feature Scaling
# Select numerical features to scale
numerical_features = ['Age', 'Fare']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("Data Info after Feature Scaling (first 5 rows):")
print(df.head())

# 8. Final Check
# The final dataset is ready for a machine learning model.
print("\nFinal Preprocessed Data Summary:")
print(df.describe())