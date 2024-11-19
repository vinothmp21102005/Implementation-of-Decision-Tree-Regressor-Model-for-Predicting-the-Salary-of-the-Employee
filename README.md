# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
step 1: Import necessary libraries like pandas, scikit-learn, and metrics.

step 2: Load employee data into a pandas DataFrame.

step 3: Divide data into features (X) and target variable (y: Salary). Split into training and testing sets.

step 4: Create a DecisionTreeRegressor object.

step 5: Train the model on the training data.

step 6: Use the trained model to predict salaries for the testing data.

 step 7: Evaluate the model's performance using metrics like MSE and MAE.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: VINOTH M P
RegisterNumber:  212223240182
*/
import pandas as pd
data=pd.read_csv("Salary.csv")
print("First five rows:\n",data.head())
data.info()
print("Null Values:\n",data.isnull().sum())

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
print("Encoded Values:\n",data.head())

x=data[["Position","Level"]]
y=data["Salary"]

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(xtrain,ytrain)
ypred=dt.predict(xtest)

from sklearn import metrics
mse=metrics.mean_squared_error(ytest,ypred)
print("Mean squared error: ",mse)

r2=metrics.r2_score(ytest,ypred)
print("R2_Score: ",r2)

print("Prediction: ",dt.predict([[5,6]]))
```

## Output:
![image](https://github.com/user-attachments/assets/a7a836e2-771e-4323-8342-6430a89ffc9f)

![image](https://github.com/user-attachments/assets/04d1a27b-bcbf-4526-99fb-b3abdfd295ed)

![image](https://github.com/user-attachments/assets/1e2632a2-f5b4-4e9b-a3d5-ffed590fffc6)

![image](https://github.com/user-attachments/assets/cca085e5-742e-4d88-8012-d03b947d29f6)

![image](https://github.com/user-attachments/assets/8c691d60-4ac1-4c73-a339-943020e42765)

![image](https://github.com/user-attachments/assets/64b0e61c-4426-4485-9d1b-38e3ce52d1b8)

![image](https://github.com/user-attachments/assets/0ff530ce-c754-4835-a527-26dc4a3c232d)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
