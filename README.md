# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start by initializing all the weights and bias to zero.
2. Calculate the predicted output for the given input data using the logistic (sigmoid) function.
3. Compare the predicted output with the actual output and compute the error.
4. Adjust the weights and bias step by step to reduce the error, and repeat this process until the model learns properly.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: PRASIDHA A 
RegisterNumber: 212224230204
*/

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("Placement_Data.csv")

data = data.drop(['sl_no', 'salary'], axis=1)

data['gender'] = data['gender'].map({'M': 1, 'F': 0})
data['workex'] = data['workex'].map({'Yes': 1, 'No': 0})
data['status'] = data['status'].map({'Placed': 1, 'Not Placed': 0})

data = pd.get_dummies(data, columns=['ssc_b', 'hsc_b', 'hsc_s', 'degree_t', 'specialisation'], drop_first=True)

X = data.drop('status', axis=1).values
y = data['status'].values.reshape(-1, 1)

X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def logistic_regression(X, y, learning_rate, iterations):
    m, n = X.shape
    w = np.zeros((n, 1))
    b = 0
    cost_list = []

    for i in range(iterations):
        z = np.dot(X, w) + b
        h = sigmoid(z)

        cost = -(1/m) * np.sum(y * np.log(h + 1e-10) + (1 - y) * np.log(1 - h + 1e-10))

        dw = (1/m) * np.dot(X.T, (h - y))
        db = (1/m) * np.sum(h - y)

        w -= learning_rate * dw
        b -= learning_rate * db

        if i % 100 == 0:
            cost_list.append(cost)
            print(f"Iteration {i} | Cost: {cost:.6f}")

    return w, b, cost_list

w, b, cost_list = logistic_regression(X, y, learning_rate=0.01, iterations=2000)

plt.plot(range(0, len(cost_list)*100, 100), cost_list)
plt.title("Cost Function Convergence")
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.show()

def predict(X, w, b):
    z = np.dot(X, w) + b
    probs = sigmoid(z)
    return [1 if i > 0.5 else 0 for i in probs]

y_pred = predict(X, w, b)
accuracy = (y_pred == y.reshape(-1)).mean() * 100
print(f"\nModel Accuracy: {accuracy:.2f}%")

```

## Output:


<img width="375" height="438" alt="Screenshot 2025-10-08 104618" src="https://github.com/user-attachments/assets/0e0b51c6-247c-4e0f-9153-a8391b86718b" />


<img width="791" height="643" alt="Screenshot 2025-10-08 104645" src="https://github.com/user-attachments/assets/aa5ee025-0302-4617-b0d0-9b44f9d932f0" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

