import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import concurrent.futures

from convexity_check import Derivatives
from util import * # utility functions for this evaluation

df = pd.read_csv("lcdb/database-accuracy.csv")

df_mean_curves = get_mean_curves(df)

r=0.5

np.seterr(all="ignore")
warnings.filterwarnings('ignore')

pool = concurrent.futures.ThreadPoolExecutor(max_workers=40)

#metrics = np.array([])

def job(df_learner, min_size, column, openmlid, learner):
    sizes = np.array(df_learner["sizes"].values[0])
    indices = np.where(sizes >= min_size)
    if np.count_nonzero(indices) < 3:
        return None
    relevant_sizes = sizes[indices]
    
    mean_accuracies = df_learner[column].values[0]
    derivative = Derivatives(x=relevant_sizes, Y=mean_accuracies, name=f'{openmlid}-{learner}')
    derivative.main()
    return [openmlid, learner, derivative.metric, derivative.inverse_metric]


def get_covexity_violations(df, column, is_increasing, min_size = 0):
    rows = []
    futures = []
    for i, (openmlid, df_dataset) in enumerate(df.groupby("openmlid")):
        
        for learner, df_learner in df_dataset.groupby("learner"):
            future = pool.submit(job, df_learner, min_size, column, openmlid, learner)
            futures.append(future)
            
            #sizes = np.array(df_learner["sizes"].values[0])
            #indices = np.where(sizes >= min_size)
            #if np.count_nonzero(indices) < 3:
            #    continue
            #relevant_sizes = sizes[indices]
            #
            #mean_accuracies = df_learner[column].values[0]
            #derivative = Derivatives(x=relevant_sizes, Y=mean_accuracies, name=f'{openmlid}-{learner}')
            #derivative.main()
#
            #rows.append([openmlid, learner, derivative.metric, derivative.inverse_metric])
            #np.append(metrics, derivative.metric)
    for i, future in enumerate(futures):
        print(f'{(100 * i / len(futures)):.2f}%')
        row = future.result()
        if row == None:
            continue
        rows.append(row)

    return pd.DataFrame(rows, columns=["openmlid", "learner", "violation", "acceptance"])

df_convexityviolations_mean = get_covexity_violations(df_mean_curves, "mean_errorrates", False)
df_convexityviolations_mean.to_csv('results/convexity_mean_violation.csv')  