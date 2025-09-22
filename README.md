## Task 1: Data Cleaning & Preprocessing
This repository contains my solution for Task 1 of the AIML  Internship, which focuses on data cleaning and preprocessing.

## Objective
The goal of this task was to learn and apply fundamental data preparation techniques on a raw dataset before it can be used for machine learning. The key steps involved were:

1. Handling missing values.

2. Converting categorical features into a numerical format.

3. Standardizing numerical features.

4. Identifying and visualizing outliers.

## Tools Used
Python: The main programming language.

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib & Seaborn: For data visualization (e.g., boxplots).

Scikit-learn: For feature scaling and imputation.

## Steps Taken
1. Data Exploration
I started by loading the Titanic-Dataset.csv file and using df.info() and df.isnull().sum() to understand the data types and the extent of missing data.

2. Handling Missing Values
Missing values in the Age column were imputed with the median.

Missing values in the Embarked column were imputed with the mode.

The Cabin column was dropped entirely due to a high number of missing values.

3. Outlier Visualization
I used a boxplot to visualize outliers in the Fare column.

4. Encoding & Scaling
The Sex and Embarked columns were converted into a numerical format using one-hot encoding, and the numerical features Age and Fare were scaled using StandardScaler.

## Repository Contents
1. data_preprocessing.py: The Python script containing all the code for data cleaning and preprocessing.

2. Titanic-Dataset.csv: The original dataset used for the task.

3. README.md: This file, which explains the project.

4. Screenshots/: This folder contains screenshots of the script's output.