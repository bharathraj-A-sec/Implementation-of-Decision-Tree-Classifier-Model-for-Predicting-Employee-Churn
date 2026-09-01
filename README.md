# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Load the Employee dataset and select employee attributes such as satisfaction level, evaluation, projects, working hours, and other relevant features.
2.Split the dataset into training and testing sets using train_test_split().
3.Create and train the Decision Tree Classifier using the training data.
4.Predict employee churn using the test data, calculate the accuracy, and predict whether a new employee is likely to leave or stay.
 ```

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Allahbakash A
RegisterNumber:  212225240007
*/
```
```
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
data = pd.read_csv("Employee (2).csv")

# Display first 5 rows
print(data.head())

# Select input features
X = data[
    [
        'satisfaction_level',
        'last_evaluation',
        'number_project',
        'average_montly_hours',
        'time_spend_company',
        'Work_accident',
        'promotion_last_5years'
    ]
]

# Target variable
y = data['left']

# Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
