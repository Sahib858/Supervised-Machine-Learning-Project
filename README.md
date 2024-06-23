# COGS118A_FINAL: Supervised Machine Learning Project

This project implements and evaluates various supervised machine learning algorithms, including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), Decision Trees, and Random Forests, on two datasets: Bankruptcy and Student Performance.

## Dataset 1: Bankruptcy

The Bankruptcy dataset contains 96 features related to various financial attributes of companies. The goal is to predict whether a company will go bankrupt or not based on these features.

### Data Preprocessing

1. Load the dataset from the `Bankrupcy.csv` file.
2. Handle missing values by imputing with the mean.
3. Drop columns with a high percentage of missing values.
4. Drop duplicate entries.
5. Detect and handle outliers using the Interquartile Range (IQR) method.
6. Separate numerical and categorical features.
7. Encode categorical features using one-hot encoding.
8. Scale numerical features using StandardScaler.
9. Perform feature selection using SelectKBest and ANOVA F-value.
10. Split the data into training and testing sets.

### Model Evaluation

1. Define helper functions for each classifier (SVM, KNN, Decision Tree, Random Forest) to perform grid search and cross-validation for hyperparameter tuning.
2. Split the data into training and testing sets with different partitions (80/20, 50/50, 20/80).
3. Train and evaluate each classifier on the different partitions.
4. Calculate and store the test accuracy, best training accuracy, and optimal hyperparameters for each classifier and partition.
5. Visualize the performance of different classifiers for bankruptcy prediction using bar plots.

## Dataset 2: Student Performance

The Student Performance dataset contains 8 features related to students' backgrounds and their performance in math, reading, and writing.

### Data Preprocessing

1. Load the dataset from the `StudentsPerformance.csv` file.
2. Handle missing values by dropping rows with missing data.
3. Identify numerical and categorical features.
4. Separate the target variable (e.g., 'math score') from the features.
5. Detect and handle outliers using z-scores.
6. Encode categorical features using a custom `MultiColumnLabelEncoder` class.
7. Scale numerical features using StandardScaler.
8. Combine encoded categorical and scaled numerical features.
9. Split the data into training and testing sets.

### Model Evaluation

1. Define helper functions for each classifier (SVM, KNN, Decision Tree, Random Forest) to perform grid search and cross-validation for hyperparameter tuning.
2. Split the data into training and testing sets with different partitions (80/20, 50/50, 20/80).
3. Train and evaluate each classifier on the different partitions.
4. Calculate and store the test accuracy, best training accuracy, and optimal hyperparameters for each classifier and partition.
5. Print the average test accuracy for each model and partition.

The project utilizes various Python libraries, including pandas, numpy, scipy, scikit-learn, matplotlib, and seaborn, for data manipulation, machine learning, and visualization tasks.
