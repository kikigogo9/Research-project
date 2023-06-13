import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from util import * # utility functions for this evaluation


df = pd.read_csv("lcdb/database-accuracy.csv")

df_mean_curves = get_mean_curves(df)

def get_covexity_violations(df, column, is_increasing, min_size = 0):
    rows = []
    for openmlid, df_dataset in tqdm(df.groupby("openmlid")):
        for learner, df_learner in df_dataset.groupby("learner"):
            sizes = np.array(df_learner["sizes"].values[0])
            indices = sizes >= min_size
            if np.count_nonzero(indices) < 3:
                continue
            relevant_sizes = sizes[indices]
            mean_scores = np.nanmean(df_learner[column].values[0], axis=0)
            #mean_scores = np.array([mean_scores])
            mean_accuracies = mean_scores[indices]
            violations = []
            for i, (a1, s1) in enumerate(zip(relevant_sizes[:-2], mean_accuracies[:-2])):
                a2 = relevant_sizes[i + 2]
                s2 = mean_accuracies[i + 2]
                A = np.vstack([[a1, a2], np.ones(2)]).T
                m, c = np.linalg.lstsq(A, [s1, s2])[0]
                required_for_convexity = m * relevant_sizes[i+1] + c
                seen = mean_accuracies[i+1]
                violations.append(max(0, seen - required_for_convexity))
            rows.append([openmlid, learner, max(violations)])
    return pd.DataFrame(rows, columns=["openmlid", "learner", "max_violation"])




df_convexityviolations_0 = get_covexity_violations(df_mean_curves, "mean_errorrates", False)
df_convexityviolations_0.to_csv('results/lcdb-convexity.csv')  