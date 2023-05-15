
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import xscale
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from tqdm import tqdm
#sorted points with regards to its x value

def calc_distances(points, A):
    distances = np.zeros(len(points))   
    for i ,p in enumerate(points):
        distances[i] = logarithmic_distance(A, p)
    return distances

def convexity(points, n=3, k=1, r=0.5):
    
    convex = []
    non_convex = []
    prev_slope = 0
    slopes = np.zeros(len(points))
    
    for i in tqdm(range(len(points))):
        distances = calc_distances(points, points[i])
        neighbors = []
        
        for j in range(len(points)):
            if distances[j] < r:
                neighbors.append(points[j])
        neighbors = np.array(neighbors)
        print(neighbors.size)
        
        X = neighbors.T[0].reshape((-1, 1))
        y = neighbors.T[1]
        model = LinearRegression()
        model.fit(X, y)

        slope = model.coef_[0]
        slopes[i] = slope

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
    
    return convex, non_convex, slopes


def logarithmic_distance(A, B): 
    return np.sqrt((np.log2(A[0]/B[0])**2))
def logarithmic_distance_2(A, B): 
    return np.abs(np.log2(A[0]/B[0])) + np.abs(A[1]-B[1])


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
            
            if points[index] in np.array(convex):
                convex_score +=1
            if points[index] in np.array(non_convex):
                non_convex_score +=1
            
        if convex_score > non_convex_score:
            res_c += [points[i]]
        if convex_score < non_convex_score:
            res_nc += [points[i]]
        if convex_score == non_convex_score:
            res_c += [points[i]]
            res_nc += [points[i]]
    
    return res_c, res_nc




def sin(x):
    r = 0
    return ((x+r)/10, np.sin((x+r)/10))

def noisy_sin(x, noise_level = 0.1):
    
    y_noise = np.random.normal(scale=noise_level)
    return (x/10, np.sin(x/10 + y_noise))


def gen():
    N = 40

    x = np.linspace(0, N, N)
    x = np.sqrt(2) ** x

    # Calculate y values using sin function
    y = 10*np.exp(-0.000005*(x-256)**2)#((x+128)*0.01)**2 * np.exp(-0.01*(x+128))#x * x * np.exp(-x) 
    scale = 0.01 * (np.max(y) - np.min(y))

    Y = y + np.random.normal(scale=scale, size=N)
     #= savgol_filter(Y, window_length=20, polyorder=2, deriv=2)
    d = (x*x-4*x+2) * np.exp(-x)
    Y = savgol_filter(Y, window_length=5, polyorder=2)
    Y = np.clip(Y, 0, None)

    return x, Y
def load():
    Y = np.loadtxt('test_std.txt')
    N = 29
    Y = -Y.flatten()
    
    
    print(N)
    x = np.linspace(0, N, N)
    x = np.sqrt(2) ** x
    x = np.repeat(x, 5)
    #Y = savgol_filter(Y, window_length=5, polyorder=2)

    return x, Y

r=1
x, Y = load()
c, nc, slopes = convexity(np.array(list(zip(x,Y))), r=r)


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

nc = np.array(nc)   


#slopes = savgol_filter(slopes, window_length=5, polyorder=2)
_, _, deriv_2 = convexity(np.array(list(zip(x,slopes))), r=r)



print(slopes)
print(deriv_2)
print(evaluate_region(1, 3, c, nc),evaluate_region(4, 20, c, nc))
#plot(list(map(lambda x: x[0], c)), list(map(lambda x: x[1], c)), 'bo', markersize=8)
#plot(list(map(lambda x: x[0], nc)), list(map(lambda x: x[1], nc)), 'ro')
plot(x, slopes)
plot(x, deriv_2)
#plot(x, d)

plot(x[::5], np.mean(Y.reshape((29,5)), axis=1))
xscale("log", base=2)
plt.legend(['dx/dy', '2nd derivative', 'learning curve'])
plt.show()
plt.hist(deriv_2) #[np.where(deriv_2 < 0)]

plt.show()

print(f'Average second derivative: {np.mean(deriv_2)}')
print(f'Average nonconvexity: {np.mean(deriv_2[np.where(deriv_2 < 0)])}')
print(f'Maximum nonconvexity: {np.max(deriv_2[np.where(deriv_2 < 0)])}')