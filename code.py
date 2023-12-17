import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

csv_path = 'Linear Regression - Sheet1.csv'
df = pd.read_csv(csv_path)

X = df['Hour'].values
Y = df['Vehicles'].values

mean_X = np.mean(X)
mean_Y = np.mean(Y)

numerator = np.sum((X - mean_X) * (Y - mean_Y))
denominator = np.sum((X - mean_X) ** 2)
m = numerator / denominator
b = mean_Y - m * mean_X

Y_pred = m * X + b

plt.scatter(X, Y, label='Actual data')
plt.plot(X, Y_pred, color='red', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression')
plt.legend()
plt.show()