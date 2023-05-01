
import numpy as np
from matplotlib.pyplot import plot
from matplotlib import pyplot as plt
#sorted points with regards to its x value
def convexity(points):
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
    
    return convex, non_convex

points = []

def sin(x):
    r = np.random.random()
    return ((x+r)/10, np.sin((x+r)/10))

def noisy_sin(x, noise_level = 0.4):
    r = np.random.random()
    y_noise = np.random.random()*noise_level - noise_level/2
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
