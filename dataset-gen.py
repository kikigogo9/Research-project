import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import learning_curve
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

def plot_learning_curve(estimator, X, y, train_sizes, cv=5, scoring='neg_mean_squared_error'):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, train_sizes=train_sizes, cv=cv, scoring=scoring)
    

    
   
    test_mean = -np.mean(test_scores, axis=1)    
    test_std = np.std(test_scores, axis=1)
    np.savetxt('test_std.txt', test_scores)
    
X, y = make_blobs(n_samples=100000, centers=27, n_features=16, random_state=12)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = SVC(kernel='linear')

N = 31        

train_sizes = np.linspace(0, N, N)
train_sizes = np.sqrt(2) ** (train_sizes-N)
print(train_sizes)      


plot_learning_curve(model, X_train, y_train, train_sizes, cv=5, scoring='neg_mean_squared_error')
