# 1. Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

try:
    df = pd.read_csv('Titanic-Dataset.csv')
except FileNotFoundError:
    print("Error: The 'Titanic-Dataset.csv' file was not found. Please make sure it's in the same directory as the script.")
    exit()

print("--- Initial Data Info ---")
df.info()
print("\nMissing values before preprocessing:")
print(df.isnull().sum())
print("-" * 30)

imputer_age = SimpleImputer(strategy='median')
df['Age'] = imputer_age.fit_transform(df[['Age']])


embarked_mode = df['Embarked'].mode()[0]
df['Embarked'] = df['Embarked'].fillna(embarked_mode)

df.drop('Cabin', axis=1, inplace=True)

print("\n--- Data after Handling Missing Values ---")
df.info()
print("\nMissing values after preprocessing:")
print(df.isnull().sum())
print("-" * 30)

plt.figure(figsize=(8, 6))
sns.boxplot(x=df['Fare'])
plt.title('Boxplot of Fare')
plt.show()

df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

print("\n--- Data after One-Hot Encoding ---")
print(df.head())
print("-" * 30)

numerical_features = ['Age', 'Fare']
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

print("\n--- Final Preprocessed Data (Scaled Features) ---")
print(df.head())
print("-" * 30)