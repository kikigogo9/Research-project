import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from convexity_check import Derivatives
from util import * # utility functions for this evaluation

df = pd.read_csv("lcdb/database-accuracy.csv")

df_mean_curves = get_mean_curves(df)

r=0.5

np.seterr(all="ignore")
warnings.filterwarnings('ignore')

#metrics = np.array([])


def get_covexity_violations(df, column, is_increasing, min_size = 0):
    rows = []
    for openmlid, df_dataset in tqdm(df.groupby("openmlid")):
        
        for learner, df_learner in df_dataset.groupby("learner"):
            sizes = np.array(df_learner["sizes"].values[0])
            indices = np.where(sizes >= min_size)
            if np.count_nonzero(indices) < 3:
                continue
            relevant_sizes = sizes[indices]
            
            mean_accuracies = df_learner[column].values[0]
            derivative = Derivatives(x=relevant_sizes, Y=mean_accuracies, r=r, name=f'{openmlid}-{learner}')
            derivative.main()

            rows.append([openmlid, learner, derivative.metric, derivative.inverse_metric])
            #np.append(metrics, derivative.metric)

    return pd.DataFrame(rows, columns=["openmlid", "learner", "violation", "acceptance"])

df_convexityviolations_mean = get_covexity_violations(df_mean_curves, "mean_errorrates", False)
df_convexityviolations_mean.to_csv('results/convexity_mean_violation.csv')  