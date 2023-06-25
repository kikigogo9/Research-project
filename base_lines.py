from tqdm import tqdm
import numpy as np
from convexity_check import Derivatives
import warnings

np.random.seed(419)
np.seterr(all="ignore")
warnings.filterwarnings('ignore')


def gen_base_line(f, name):
    N=20
    count = 0
    means = np.zeros((100,2))
    for j in tqdm(range(100)):
        Ys = np.zeros((125,N))
        x = np.linspace(1, N, N)
        x = np.sqrt(2) ** x * 16
        for i in range(125):
            Y = f(x)
            Ys[i] = Y + np.random.normal(0,0.05*(np.max(Y) - np.min(Y)),N)
        derivative = Derivatives(x=x, Y=Ys, draw_result=True)
        derivative.preprocess()
        derivative.map()
        derivative.slope()
        derivative_2 = Derivatives(x=derivative.x, Y=derivative.slopes, N=derivative.N, name=name, draw_result=True)
        deriv_2 = derivative_2.slope()
        derivative_2.save_image(derivative.Y.reshape((20, 125)).T)
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


def sqr(x):
    return x**2

def n_sqr(x):
    return -x**2

def exp_hill(x):
    return np.exp(-0.05 * x) + 0.4 * np.exp(-(x-128)**2/16)

gen_base_line(sqr, "000-ground-up")

gen_base_line(n_sqr, "000-ground-down")

gen_base_line(exp_hill, "000-hill")