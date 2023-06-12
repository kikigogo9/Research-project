import numpy as np
import pandas as pd
import itertools as it
import matplotlib.pyplot as plt


df = pd.read_csv('results/convexity_mean_violation.csv')
df["M"] = -df["violation"] + df["acceptance"]
df = df.sort_values(by=['M'])


#print(df)
"""

"""
#print(df[df['M'] < 0].sort_values(by=['violation']))
#print(df[df['M'] >= 0].sort_values(by=['acceptance'], ascending=False))

df_learner = df.groupby(['learner']).count().sort_values(by=['Unnamed: 0'],ascending=False)
df_dataset = df.groupby(['openmlid']).count().sort_values(by=['Unnamed: 0'],ascending=False)

df_v_learner = df[df['M'] < 0].groupby(['learner']).count().sort_values(by=['Unnamed: 0'],ascending=False)
df_a_learner = df[df['M'] >= 0].groupby(['learner']).count().sort_values(by=['Unnamed: 0'],ascending=False)      



df_v_dataset = df[df['M'] < 0].groupby(['openmlid']).count().sort_values(by=['Unnamed: 0'],ascending=False)
df_a_dataset = df[df['M'] >= 0].groupby(['openmlid']).count().sort_values(by=['Unnamed: 0'],ascending=False)      


df_learner['count'] = (df_v_learner['Unnamed: 0'])/df_learner['Unnamed: 0']
df_dataset['count'] = df_v_dataset['Unnamed: 0']/df_dataset['Unnamed: 0']

df_learner = df_learner.replace(np.nan,0)
df_dataset = df_dataset.replace(np.nan,0)

print('-------Experiment 1--------')

print(df[df['M'] < 0].count()/df.count())
print(df_learner.sort_values(by=['count'], ascending=False)['count'])
print(df_dataset.sort_values(by=['count'])['count'])


print('-------Experiment 2--------')


df_2 = pd.read_csv('results/convexity_single_violation.csv')

df_2["M"] = (-df_2["violation"] + df_2["acceptance"]) 
df_2["K"] = df["M"]
df_2["L"] = df_2["M"] * df_2['K']
df_2 = df_2.sort_values(by=['L'])

df_2_learner = df_2[df_2['L'] < 0].groupby(['learner']).count().sort_values(by=['Unnamed: 0'],ascending=False)
df_2_learner['count'] = df_2_learner['Unnamed: 0']/ df_2.groupby(['learner']).count()['Unnamed: 0']

df_2_nonconvex = df_2[df_2['K'] < 0][df_2['L'] < 0].count()
df_2_nonconvex['count'] = df_2_nonconvex['Unnamed: 0']/ df_2[df_2['K'] < 0].count()['Unnamed: 0']
df_2_convex = df_2[df_2['K'] >= 0][df_2['L'] < 0].count()
df_2_convex['count'] = df_2_convex['Unnamed: 0']/ df_2[df_2['K'] >= 0].count()['Unnamed: 0']
#percentage of curves, where the curve was identified as convex/nonconvex, but the convexity of the individual curves was the opposite
print(df_2_nonconvex['count'] , df_2_convex['count'])
print(df_2_learner.sort_values(by=['count'])['count'])

print('--------Experiment 3--------')
df_confidence_intervall =  pd.read_csv('results/confidence_interval.csv')

df_ci = df_confidence_intervall.groupby(['learner']).mean().sort_values(by=['metric'],ascending=False)
print(df_ci['metric'])
print(df_confidence_intervall.sort_values(by=['metric'],ascending=False))