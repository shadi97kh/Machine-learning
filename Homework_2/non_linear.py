import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Loading and reading each line of the dataset
data = pd.read_csv('HW2_nonlinear_data.csv')
X = data.iloc[:, 0].values
Y = data.iloc[:, 1].values

# This function will compute our loss function
def compute_mse(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()

# This function constantly compute the gradients and update the parameters
def gradient_descent_cubic(X, Y, learning_rate, epochs):
    a, b, c, d = 0, 0, 0, 0
    n = len(X)

    for _ in range(epochs):
        Y_pred = a*X**3 + b*X**2 + c*X + d
        mse = compute_mse(Y, Y_pred)

        # Compute gradients
        da = (-2/n) * sum(X**3 * (Y - Y_pred))
        db = (-2/n) * sum(X**2 * (Y - Y_pred))
        dc = (-2/n) * sum(X * (Y - Y_pred))
        dd = (-2/n) * sum(Y - Y_pred)

        # Update parameters
        a = a - learning_rate * da
        b = b - learning_rate * db
        c = c - learning_rate * dc
        d = d - learning_rate * dd

    return a, b, c, d
    
    
learning_rate = 1e-6
epochs = 10000
a, b, c, d = gradient_descent_cubic(X, Y, learning_rate, epochs)

# Plotting the actual points
plt.scatter(X, Y, color='blue', marker='o', label='Data Points')

# Predicting the Y values using the trained model
Y_pred = a*X**3 + b*X**2 + c*X + d

# Plotting the regression results using dots
plt.scatter(X, Y_pred, color='red', marker='.', label='Regression Dots')

plt.title('Cubic Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()



