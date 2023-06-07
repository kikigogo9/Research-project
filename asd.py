import numpy as np
import statistics as stats
# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 5, 9, 15, 22])

# Fit polynomial regression
degree = 2
coefficients = np.polyfit(x, y, degree)
# Compute residuals
predicted_values = np.polyval(coefficients, x)
residuals = y - predicted_values
# Compute standard deviation of residuals
rmse = np.sqrt(np.sum(residuals**2) / (len(x) - degree - 1))
# Compute standard errors for coefficients
x_mean = np.mean(x)
X = np.vander(x, degree + 1, increasing=True)
X[:, 1:] *= np.power.outer(x - x_mean, np.arange(1, degree + 1))
XtX_inv = np.linalg.inv(np.dot(X.T, X))
coeff_std_errors = np.sqrt(np.diagonal(rmse**2 * XtX_inv))
# Set the desired confidence level
confidence_level = 0.95

# Calculate the t-value
t_value = np.abs(np.round(stats.t.ppf((1 - confidence_level) / 2, len(x) - degree - 1), decimals=4))
# Calculate confidence intervals for coefficients
lower_bounds = coefficients - t_value * coeff_std_errors
upper_bounds = coefficients + t_value * coeff_std_errors

# Combine lower and upper bounds
confidence_interval = np.vstack((lower_bounds, upper_bounds)).T
