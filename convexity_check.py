
import numpy as np
from matplotlib.pyplot import plot
from matplotlib.pyplot import xscale, yscale
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.signal import savgol_filter
from tqdm import tqdm
from scipy.stats import t
#sorted points with regards to its x value

IMAGE_COUNT = 0

class Derivatives:
    def __init__(self, x=[], Y=[], r=0.5, name='default', N=1, mode='all', draw_result=False):
        self.x = x
        self.Y = Y
        self.r = r
        #self.points = np.array(list(zip(x,Y)))
        self.slopes = None
        self.confidence_interval = None
        self.name = name
        self.metric = 0.0
        self.inverse_metric = 0.0
        self.N=N
        self.mode = mode
        self.alpha = 0.95
        if (mode == 'all'):
            self.slope = self.slope_all
        elif (mode == 'single'):
            self.slope = self.slope_single

        self.draw_image = draw_result
        self.debug = False

    def calc_distances(self, A):
        distances = np.zeros(len(self.points))   
        for i ,p in enumerate(self.points):
            distances[i] = self.logarithmic_distance(A, p)
        return distances

    def slope_single(self):
        
        self.slopes = np.zeros(len(self.x))


        for i in range(len(self.x)):
            low = max(i % self.N, i - self.N) 
            high = i + 2*self.N
            #neighbors = self.points[low:high]
            #print(i // self.N * self.N , i // self.N * self.N + 2 * self.N)
            

            X = self.x[low:high:self.N].reshape((-1, 1))
            y = self.Y[low:high:self.N]
            model = LinearRegression()
            model.fit(X, y)
            


            self.slopes[i] = model.coef_[0]
        
        return self.slopes
        
    def slope_all(self):
        
        self.slopes = np.zeros(len(self.x))
        self.confidence_interval = np.zeros((len(self.x), 2))
        
        for i in range(len(self.x)):
            low = max(0, (i//self.N -1) * self.N)
            high = (i // self.N +2) * self.N
            #neighbors = self.points[low:high]
            #print(i // self.N * self.N , i // self.N * self.N + 2 * self.N)
            

            X = self.x[low:high].reshape((-1, 1))
            y = self.Y[low:high]
            model = LinearRegression()
            model.fit(X, y)

            o_2 = ((model.predict(X) - y) ** 2 / (len(y) - 2)).sum()
            SE =  np.sqrt(o_2/((np.mean(X)-X)**2).sum())
            critical_value = t.ppf((1 - self.alpha) / 2, len(X)-1)
            interval = SE * critical_value

            
            self.slopes[i] = model.coef_[0]
            self.confidence_interval[i] = np.array([model.coef_[0]-interval, model.coef_[0]+interval])
            
        
        return self.slopes

    def default_derivative_calculator(self):
        dx = np.diff(self.x)
        dy = np.diff(self.Y)
    

        dydx = dy/dx

        return self.x[1:], dydx


    def logarithmic_distance(self, A, B): 
        return np.sqrt((np.log2(A[0]/B[0])**2))
    def logarithmic_distance_2(self,A, B): 
        return np.abs(np.log2(A[0]/B[0])) 

    def gen(self):
        N = 40

        x = np.linspace(0, N, N)
        x = np.sqrt(2) ** x

        # Calculate y values using sin function
        y = 10*np.exp(-0.000005*(x-256)**2)#((x+128)*0.01)**2 * np.exp(-0.01*(x+128))#x * x * np.exp(-x) 
        scale = 0.01 * (np.max(y) - np.min(y))

        Y = y + np.random.normal(scale=scale, size=N)
         #= savgol_filter(Y, window_length=20, polyorder=2, deriv=2)
        #d = (x*x-4*x+2) * np.exp(-x)
        #Y = savgol_filter(Y, window_length=5, polyorder=2)
        #Y = np.clip(Y, 0, None)

        self.x, self.Y = x, Y 
        
    def preprocess(self):
        self.N = self.Y.shape[0]
        
        self.Y = self.Y.T.flatten()
        
    
        self.x = np.repeat(self.x, self.N)

        #self.points = np.array(list(zip(self.x, self.Y)))
        #self.Y = savgol_filter(self.Y, window_length=self.N*2, polyorder=2)
        return self.x, self.Y
    
    def map(self):
        if (np.max(self.Y) - np.min(self.Y)) != 0:
            self.Y = (self.Y - np.min(self.Y)) /  (np.max(self.Y) - np.min(self.Y))
            #print(np.max(self.Y) , np.min(self.Y))
        #self.points = np.column_stack((self.x,self.Y))


    def save_image(self, Y):
        
        if self.draw_image:
            plt.rcParams.update({'font.size': 22})
            x = self.x[::self.N]
            deriv_1 = self.Y[::self.N]
            deriv_2 = self.slopes[::self.N]
            Y_mean = np.mean(Y, axis=0)

            fig, ax = plt.subplots(2, 2, figsize=(25, 16))

            
            
            nonconvex = deriv_2[np.where(deriv_2 < 0)]
            convex = deriv_2[np.where(deriv_2 >= 0)]
            
            bins=np.histogram(np.hstack((nonconvex,convex)), bins=40)[1] #get the bin edges

            if len(convex) > 0:
                ax[0,0].hist(convex, bins, linewidth=1) #[np.where(deriv_2 < 0)]
            if len(nonconvex) > 0:    
                ax[0,0].hist(nonconvex, bins, linewidth=1) #[np.where(deriv_2 < 0)]

            #plt.hist(d2[np.where(d2 >= 0)], hist) #[np.where(deriv_2 < 0)]
            #plt.hist(d2[np.where(d2 < 0)], hist) #[np.where(deriv_2 < 0)]
            ax[0,0].legend(['number of convex anchors', 'number of nonconvex anchors'])
            ax[0,0].set_xlabel('Magnitude of Second Derivatives')
            ax[0,0].set_ylabel('Second Derivative Value Frequency')
            ax[0,0].set_title('Frequency Distribution of Second Derivative Values')

            ax[0,1].set_xscale("log", base=2)
            ax[1,1].set_xscale("log", base=2)
            ax[1,0].set_xscale("log", base=2)
            ax[1,0].plot(x, deriv_1, color='green')
            ax[1,1].plot(x, deriv_2, color='red')
            ax[0,1].plot(x, Y_mean)
            ax[0,1].fill_between(x, np.max(Y, axis=0), np.min(Y, axis=0), alpha=0.3)
            #for i in range(5):
            #    plot(x[i::5], Y[i::5])
            #        xscale("log", base=2)
            ax[0,1].legend(['learning curve'])
            ax[1,1].legend(['2nd derivative'])
            ax[1,0].legend(['1st derivative'])
            ax[0,1].set_xlabel('Number of training samples')
            ax[0,1].set_ylabel('Error rate')
            ax[0,1].set_title('Learning Curve and its Derivatives')


#            fig.tight_layout()
        
        
            plt.savefig(f'results/plots/{self.name}.png')

        if self.debug:
            print(f'Average second derivative: {np.mean(deriv_2)}')
            print(f'Average nonconvexity: {np.mean(deriv_2[np.where(deriv_2 < 0)])}')
            print(f'Maximum nonconvexity: {np.min(deriv_2[np.where(deriv_2 < 0)])}')
        #np.sum(self.slopes[np.where(self.slopes < 0)]) *
        #np.sum(self.slopes[np.where(self.slopes >= 0)]) *
        if self.mode == 'all':
            self.metric =  len(self.slopes[np.where(self.slopes < 0)]) / len(self.slopes)#len(self.slopes[np.where(self.slopes < 0)]) / len(self.slopes)#
            self.inverse_metric = len(self.slopes[np.where(self.slopes >= 0)]) / len(self.slopes)
            if np.isnan(self.metric):
                self.metric = 0.0
            if np.isnan(self.inverse_metric):
                self.inverse_metric = 0.0

        else:
            self.metric = np.zeros(self.N)
            self.inverse_metric = np.zeros(self.N)
            for i in range(self.N):
                slopes = self.slopes[i::self.N]
                self.metric[i] = len(slopes[np.where(slopes < 0)]) / len(slopes)#len(self.slopes[np.where(self.slopes < 0)]) / len(self.slopes)#
                self.inverse_metric[i] =  len(slopes[np.where(slopes >= 0)]) / len(slopes)
            
            self.metric = -self.metric + self.inverse_metric
            
            self.inverse_metric = len(self.metric[np.where(self.metric >= 0)])/len(self.metric)
            self.metric = len(self.metric[np.where(self.metric < 0)])/len(self.metric)
        #print(self.slopes, len(self.slopes[np.where(self.slopes >= 0)]))


        plt.close()
        #plt.show()
    def getConfidence(self):
        Y = np.array(self.Y, copy=True)  
        #preprocess anchorns
        self.preprocess()
        self.map()
        #Calculate dy/dx
        self.slope()
        derivative = Derivatives(x=self.x, Y=self.slopes, r=self.r, name=self.name, N=self.N, mode='all')
        #Calculate d2y/dx2
        derivative.slope()
        derivative.save_image(Y)

        count = 0
        for i, slope in enumerate(derivative.slopes):
            
            if slope > 0 and (derivative.confidence_interval[i, 1] < 0):
                count +=1
            if slope < 0 and (derivative.confidence_interval[i, 0] > 0):
                count +=1

        self.metric = count/len(derivative.slopes) 


    def main(self):
        Y = np.array(self.Y, copy=True)  
        #preprocess anchorns
        self.preprocess()
        self.map()
        #Calculate dy/dx
        self.slope()
        derivative = Derivatives(x=self.x, Y=self.slopes, r=self.r, name=self.name, N=self.N, mode=self.mode)
        #Calculate d2y/dx2
        derivative.slope()
        derivative.save_image(Y)

        self.metric = derivative.metric 
        self.inverse_metric = derivative.inverse_metric 


if __name__ == "__main__":
    #np.random.seed(420)
    N=20
    count = 0
    means = np.zeros((100,2))
    for j in tqdm(range(100)):
        Ys = np.zeros((125,N))
        x = np.linspace(1, N, N)
        x = np.sqrt(2) ** x * 16
        for i in range(125):
            Y = x ** 2
            Ys[i] = Y + np.random.normal(0,0.05*(np.max(Y) - np.min(Y)),N)
        derivative = Derivatives(x=x, Y=Ys, draw_result=True)
        derivative.preprocess()
        derivative.map()
        derivative.slope()
        derivative_2 = Derivatives(x=derivative.x, Y=derivative.slopes, N=derivative.N, name="000-ground-up", draw_result=True)
        deriv_2 = derivative_2.slope()
        derivative_2.save_image(Ys)
        derivative.metric = derivative_2.metric 
        derivative.inverse_metric = derivative_2.inverse_metric 
        #plot(x,derivative.Y)
        #xscale('log', base=2)
        #plt.show()
        #print(deriv_2)
        #print(derivative.metric, derivative.inverse_metric)

        means[j,0] = derivative_2.metric
        means[j,1] = derivative_2.inverse_metric
    means = means.T[1]-means.T[0]
    print(np.mean(means, axis=0), np.min(means, axis=0), np.max(means, axis=0))


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
#deriv_2 = deriv_2[10::5]

#hist = [-0.05,-0.025, 0, 0.025, 0.05, 0.15]
#plt.hist(deriv_2[np.where(deriv_2 >= 0)], hist) #[np.where(deriv_2 < 0)]
#plt.hist(deriv_2[np.where(deriv_2 < 0)], hist) #[np.where(deriv_2 < 0)]

#plt.hist(d2[np.where(d2 >= 0)], hist) #[np.where(deriv_2 < 0)]
#plt.hist(d2[np.where(d2 < 0)], hist) #[np.where(deriv_2 < 0)]
#plt.legend(['number of convex anchors', 'number of nonconvex anchors'])
#plt.xlabel('Magnitude of the Second Derivative')
#plt.ylabel('Second derivative value frequency')
#plt.title('Frequency Distribution of Second Derivative Values')
#plt.show()


