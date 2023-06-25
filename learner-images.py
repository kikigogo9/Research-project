import numpy as np
from scipy.stats import t
from matplotlib import pyplot as plt

import matplotlib 



array = {
"QuadraticDiscriminantAnalysis":   0.194175,
"SVC_sigmoid":    0.283401,
"PassiveAggressiveClassifier":    0.177419,
"Perceptron":    0.133065,
"SGDClassifier":    0.129032,
"SVC_linear":    0.092742,
"BernoulliNB":    0.108911,
"SVC_rbf":     0.076613,
"MLPClassifier":    0.07804,
"SVC_poly":    0.048387,
"LogisticRegression":    0.061475,
"MultinomialNB":    0.066176,
"KNeighborsClassifier":     0.044118,
"GradientBoostingClassifier":    0.036585,
"RidgeClassifier":    0.043269,
"RandomForestClassifier":    0.034653,
"ExtraTreeClassifier":    0.024876,
"DecisionTreeClassifier":    0.024752,
"ExtraTreesClassifier":    0.024631,
"LinearDiscriminantAnalysis":    0.011628}

matplotlib.rcParams.update({'font.size': 18})

x = np.load('out.npz.npy')
array = dict(sorted(array.items(), key=lambda item: item[1], reverse=True))
print(array)
fig, ax = plt.subplots(1, figsize = (12, 8))
plt.xticks(rotation = 75)
plt.bar(list(array.keys()), array.values())
#plt.hist(x)
plt.xticks(rotation=40, ha='right')
plt.ylabel("Noncovex curve fraction")
plt.title("Fraction of Learners with Nonconvex Learning Curves")
plt.tight_layout()
plt.show()



array = {
    "871" :   0.473684,
    "42733" :   0.500000,
    "4137" :   0.500000,
    "1112" :   0.545455,
    "1114" :   0.636364
}

array = dict(sorted(array.items(), key=lambda item: item[1], reverse=True))
print(array)
fig, ax = plt.subplots(1, figsize = (12, 8))
plt.xticks(rotation = 75)
plt.bar(list(array.keys()), array.values())
#plt.hist(x)
plt.xticks(rotation=40, ha='right')
plt.xlabel("OpenmlId")
plt.ylabel("Noncovex curve fraction")
plt.title("Top 5 datasets with the highest nonconvex curve fraction")
plt.tight_layout()
plt.show()