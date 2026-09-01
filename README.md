# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the employee dataset and convert categorical data into numerical values.
2. Separate the input features and employee churn target.
3. Split the data into training and testing sets, then train the Decision Tree Classifier using Gini criterion.
4. Visualize the decision tree to show the employee churn classification. 

## Program:
#### Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
#### Developed by: ANISH ADAN THIVAKARAN
RegisterNumber: 212225230017
```
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
data = pd.read_csv("Employee.csv")
data = pd.get_dummies(data, drop_first=True)
X = data.iloc[:, :-1]   
y = data.iloc[:, -1]    
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = DecisionTreeClassifier(criterion='gini', random_state=42)
model.fit(X_train, y_train)
plt.figure(figsize=(25,12))
plot_tree(
    model,
    feature_names=X.columns,
    class_names=[str(i) for i in model.classes_],
    filled=True
)
plt.title("Decision Tree Classifier")
plt.show()
```

## Output:
<img width="1376" height="697" alt="image" src="https://github.com/user-attachments/assets/dfae640d-4d6b-452e-8415-08ca6797bb05" />

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
