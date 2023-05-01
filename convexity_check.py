
import numpy as np
from matplotlib.pyplot import plot
from matplotlib import pyplot as plt
#sorted points with regards to its x value
def convexity(points, n=4, k=3):
    convex = []
    non_convex = []
    prev_slope = 0
    for i in range(len(points)-1):
        slope = (points[i+1][1] - points[i][1])/(points[i+1][0] - points[i][0]) 
        if slope > prev_slope:
            convex += [points[i]]
        if slope < prev_slope:
            non_convex += [points[i]]
        if slope == prev_slope:
            convex += [points[i]]
            non_convex += [points[i]]
        prev_slope = slope
    # Check wether in the nearest n points are more convex than nonconvex points
    for i in range(k):
        convex, non_convex = renormalize(points, convex, non_convex, n=n)
    
    return convex, non_convex

def renormalize(points, convex, non_convex, n=4):
    res_c = []
    res_nc = []

    for i in range(len(points)):
        convex_score = 0
        non_convex_score = 0
        for j in range(-n//2, n//2+1):
            index = min(max(0, i+j), len(points)-1)
            if points[index] in convex:
                convex_score +=1
            if points[index] in non_convex:
                non_convex_score +=1
            
        if convex_score > non_convex_score:
            res_c += [points[i]]
        if convex_score < non_convex_score:
            res_nc += [points[i]]
        if convex_score == non_convex_score:
            res_c += [points[i]]
            res_nc += [points[i]]
    
    return res_c, res_nc


points = []

def sin(x):
    r = np.random.random()
    return ((x+r)/10, np.sin((x+r)/10))

def noisy_sin(x, noise_level = 0.1):
    r = np.random.random()
    y_noise = np.random.normal(scale=noise_level)
    return ((x+r)/10, np.sin((x+r)/10 + y_noise))



for i in range(100):
    points += [sin(i)]


c, nc = convexity(points)

x = np.linspace(0, 10, 100)

# Calculate y values using sin function
y = np.sin(x)


print(c, nc)
plot(list(map(lambda x: x[0], c)), list(map(lambda x: x[1], c)), 'bo')
plot(list(map(lambda x: x[0], nc)), list(map(lambda x: x[1], nc)), 'ro')
plot(x, y, 'g')
plt.show()
