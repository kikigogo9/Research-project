
import numpy as np
from matplotlib.pyplot import plot
from matplotlib import pyplot as plt
#sorted points with regards to its x value
def convexity(points, n=2, k=1):
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
            index = i+j
            if index < 0 or index >= len(points):
                continue
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
    r = 0
    return ((x+r)/10, np.sin((x+r)/10))

def noisy_sin(x, noise_level = 0.1):
    
    y_noise = np.random.normal(scale=noise_level)
    return (x/10, np.sin(x/10 + y_noise))



for i in range(100):
    points += [noisy_sin(i)]


c, nc = convexity(points)

x = np.linspace(0, 10, 100)

# Calculate y values using sin function
y = np.sin(x)

def evaluate_region(start, stop, convex, non_convex):
    c = 0
    nc = 0
    for p in convex:
        if p[0] >= start and p[0] < stop:
            c +=1

    for p in non_convex:
        if p[0] >= start and p[0] < stop:
            nc +=1
    return c, nc


print(evaluate_region(0, 10, c, nc))
plot(list(map(lambda x: x[0], c)), list(map(lambda x: x[1], c)), 'bo')
plot(list(map(lambda x: x[0], nc)), list(map(lambda x: x[1], nc)), 'ro')
plot(x, y, 'g')
plt.show()
