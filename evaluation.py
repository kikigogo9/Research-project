import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt


df = pd.read_csv('results/convexity_mean_violation.csv')
df["M"] = -df["violation"] + df["acceptance"]
df = df.sort_values(by=['M'])


print(df)
print(df[df['M'] < 0].sort_values(by=['violation']))
print(df[df['M'] >= 0].sort_values(by=['acceptance']))