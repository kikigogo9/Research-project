
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import xscale, yscale
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from tqdm import tqdm
#sorted points with regards to its x value

class Derivatives:
    def __init__(self, x=[], Y=[], r=1.0):
        self.x = x
        self.Y = Y
        self.r = r
        self.points = np.array(list(zip(x, Y)))


    def calc_distances(self, A):
        distances = np.zeros(len(self.points))   
        for i ,p in enumerate(self.points):
            distances[i] = self.logarithmic_distance(A, p)
        return distances

    def slope(self):
        slopes = np.zeros(len(self.points))

        for i in tqdm(range(len(self.points))):
            distances = self.calc_distances(self.points[i])
            neighbors = []
        
            for j in range(len(self.points)):
                if distances[j] <= self.r:
                    neighbors.append(self.points[j])
            neighbors = np.array(neighbors)
            print(neighbors.size)

            X = neighbors.T[0].reshape((-1, 1))
            y = neighbors.T[1]
            model = LinearRegression()
            model.fit(X, y)

            slopes[i] = model.coef_[0]
        
        return slopes

    def default_derivative_calculator(self):
        dx = np.diff(self.x)
        dy = np.diff(self.Y)
    

        dydx = dy/dx

        return self.x[1:], dydx


    def logarithmic_distance(self, A, B): 
        return np.sqrt((np.log2(A[0]/B[0])**2))
    def logarithmic_distance_2(self,A, B): 
        return np.abs(np.log2(A[0]/B[0])) + np.abs(A[1]-B[1])








    def gen(self):
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

        self.x, self.Y = x, Y 
        
    def load(self):
        self.Y = np.loadtxt('test_std.txt')
        N = self.Y.shape[0]
        self.Y = 1.0-self.Y.flatten()
    
    
        print(N)
        self.x = np.linspace(0, N, N)
        self.x = np.sqrt(2) ** self.x
        self.x = np.repeat(self.x, 5)

        self.points = np.array(list(zip(self.x, self.Y)))
        #Y = savgol_filter(Y, window_length=5, polyorder=2)
        return self.x, self.Y


r=1.0

derivative = Derivatives(r=r)
x, Y = derivative.load()
slopes = derivative.slope()

derivative = Derivatives(x=derivative.x, Y=slopes, r=r)

deriv_2 = derivative.slope()

print(deriv_2)
#x2, d1 = default_derivative_calculator(x[::5], np.mean(Y.reshape((29,5)), axis=1))
#x2, d2 = default_derivative_calculator(x2, d1)

#print(slopes)
#Y = Y[15:]
#slopes = slopes[15:]
#deriv_2 = deriv_2[15:]
#x = x[15:]
#print(deriv_2)
#plot(list(map(lambda x: x[0], c)), list(map(lambda x: x[1], c)), 'bo', markersize=8)
#plot(list(map(lambda x: x[0], nc)), list(map(lambda x: x[1], nc)), 'ro')
#plot(x, slopes)
#plot(x, deriv_2)
#plot(x2, d2)
#plot(x, d)

#plot(x[::5], np.mean(Y.reshape((27,5)), axis=1))
#for i in range(5):
#    plot(x[i::5], Y[i::5])

#xscale("log", base=2)
#plt.legend(['1st derivative', '2nd derivative', 'learning curve'])
#plt.xlabel('Number of training samples')
#plt.ylabel('Error rate')
#plt.title('Learning Curve and its Derivatives')
#plt.show()
deriv_2 = deriv_2[10::5]

hist = [-0.05,-0.025, 0, 0.025, 0.05, 0.15]
plt.hist(deriv_2[np.where(deriv_2 >= 0)], hist) #[np.where(deriv_2 < 0)]
plt.hist(deriv_2[np.where(deriv_2 < 0)], hist) #[np.where(deriv_2 < 0)]

#plt.hist(d2[np.where(d2 >= 0)], hist) #[np.where(deriv_2 < 0)]
#plt.hist(d2[np.where(d2 < 0)], hist) #[np.where(deriv_2 < 0)]
plt.legend(['number of convex anchors', 'number of nonconvex anchors'])
plt.xlabel('Error rate')
plt.ylabel('Second derivative value frequency')
plt.title('Frequency Distribution of Second Derivative Values')
plt.show()


print(f'Average second derivative: {np.mean(deriv_2)}')
print(f'Average nonconvexity: {np.mean(deriv_2[np.where(deriv_2 < 0)])}')
print(f'Maximum nonconvexity: {np.min(deriv_2[np.where(deriv_2 < 0)])}')