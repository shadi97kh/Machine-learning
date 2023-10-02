import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#loading and reading each line of a dataset
data = pd.read_csv('HW2_linear_data.csv')
X = data.iloc[:,0].values
Y = data.iloc[:,1].values


# This function will compute our loss function
def compute_mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

def gradient_descent(X, Y, learning_rate, epochs):
    m = 0
    c = 0
    n = len(X)
    
    for _ in range(epochs):
        Y_pred = m*X + c
        mse = compute_mse(Y, Y_pred)
        
        #compute the gradient
        dm = (-2/n) * sum(X * (Y - Y_pred))
        dc = (-2/n) * sum(Y - Y_pred)
        
        #update the parameters
        m = m - learning_rate *dm
        c = c - learning_rate *dc
        
    return m, c


learning_rate = 0.0001
epochs = 1000
m, c = gradient_descent(X, Y, learning_rate, epochs)

# Plotting the actual points
plt.scatter(X, Y, color='red', marker='o', label='Data Points')

# Predicting the Y values using the trained model
Y_pred = m*X + c

# Plotting the regression line
plt.plot(X, Y_pred, color='blue', label='Regression Line')

plt.title('Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()
