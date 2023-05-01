import numpy as np
from sklearn.linear_model import LinearRegression

# Generate random data
n = 100
x = np.linspace(0, 1, n)
y = 2 * x + np.random.normal(size=n)

# Reshape the data to fit the expected input shape for the LinearRegression model
X = x.reshape((-1, 1))
print(y)
# Fit a linear regression model to the data
model = LinearRegression()
model.fit(X, y)

# Get the slope and intercept of the line
slope = model.coef_[0]
intercept = model.intercept_

# Print the results
print(f"Slope: {slope:.2f}")
print(f"Intercept: {intercept:.2f}")