import numpy as np
import matplotlib.pyplot as plt

# Generate random points with noise
N = 20
X = np.linspace(-1, 1, N)
Y = np.sin(np.pi * X) + np.random.normal(scale=0.1, size=N)
points = np.vstack((X, Y)).T

# Define a function to check if a point is convex with respect to its closest neighbors
def is_convex(points, index, k):
    neighbors = points[np.argsort(np.linalg.norm(points - points[index], axis=1)), :]
    n = len(neighbors)
    p1 = neighbors[(index - 1) % n]
    p2 = neighbors[index]
    p3 = neighbors[(index + 1) % n]
    cross_product = np.cross(p2 - p1, p3 - p2)
    return cross_product >= 0

# Identify the non-convex points
nonconvex_inliers = []
k = 2  # number of neighbors to consider
for i in range(N):
    if not is_convex(points, i, k):
        nonconvex_inliers.append(i)

# Plot the set of points
plt.scatter(points[:, 0], points[:, 1])

# Plot the non-convex points in blue
nonconvex_inliers = np.array(nonconvex_inliers)
plt.scatter(points[nonconvex_inliers, 0], points[nonconvex_inliers, 1], c='b', s=100)

# Add labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Non-convex points with respect to closest neighbors')

# Show the plot
plt.show()