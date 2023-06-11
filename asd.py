import numpy as np
from scipy.stats import t
from sklearn.linear_model import LinearRegression

# Sample data
X = np.array([1, 2, 3, 4, 5]).reshape((-1,1))
y = np.array([2, 5, 9, 15, 22])
alpha = 0.05
model = LinearRegression()
model.fit(X, y)

o_2 = ((model.predict(X) - y) ** 2 / (len(y) - 2)).sum()
SE =  np.sqrt(o_2/((np.mean(X)-X)**2).sum())
critical_value = t.ppf(1 - alpha / 2, len(X)-1)
interval = SE * critical_value

    
asd = np.array([model.coef_[0]-interval, model.coef_[0]+interval])
print(model.coef_[0])
print(asd)