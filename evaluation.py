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

print(df_v_dataset['Unnamed: 0'])
print(df_learner.sort_values(by=['count'], ascending=False)['count'])
print(df_dataset.sort_values(by=['count'])['count'])