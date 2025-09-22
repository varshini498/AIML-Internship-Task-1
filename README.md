# 📊 Task 1: Data Cleaning & Preprocessing

This repository contains my solution for Task 1 of the A.I./M.L. Internship, which focuses on data cleaning and preprocessing.

## 📝 Objective

The goal of this task was to learn and apply fundamental data preparation techniques on a raw dataset before it can be used for machine learning. The key steps involved were:
* Handling missing values.
* Converting categorical features into a numerical format.
* Standardizing numerical features.
* Identifying and visualizing outliers.

## ⚙️ Tools Used

* **Python**: The main programming language.
* **Pandas**: For data manipulation and analysis.
* **NumPy**: For numerical operations.
* **Matplotlib & Seaborn**: For data visualization (e.g., boxplots).
* **Scikit-learn**: For feature scaling and imputation.

## 🚀 Steps Taken

### 1. Data Exploration
I started by loading the `Titanic-Dataset.csv` file and using `df.info()` and `df.isnull().sum()` to understand the data types and the extent of missing data.
(Screenshots/initial_data_info.png)

### 2. Handling Missing Values
* Missing values in the `Age` column were imputed with the **median**.
* Missing values in the `Embarked` column were imputed with the **mode**.
* The `Cabin` column was dropped entirely due to a high number of missing values.

### 3. Encoding Categorical Data
The `Sex` and `Embarked` columns were converted into a numerical format using **one-hot encoding** with `pd.get_dummies()`.

### 4. Feature Scaling
The numerical features `Age` and `Fare` were scaled using **`StandardScaler`** from scikit-learn to ensure they have a mean of 0 and a standard deviation of 1.

### 5. Outlier Visualization
I used a boxplot to visualize outliers in the `Fare` column. 
(Screenshots/fare_boxplot.png)

### 6. Final Preprocessed Data
After all the steps, the data is ready for a machine learning model.
(Screenshots/final_preprocessed_data.png)

## 📦 Repository Contents

* `data_preprocessing.py`: The Python script containing all the code for data cleaning and preprocessing.
* `Titanic-Dataset.csv`: The original dataset used for the task.
* `README.md`: This file, which explains the project.

---
_Feel free to contact me if you have any questions._