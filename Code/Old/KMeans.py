# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 16:27:49 2020

@author: Yann
"""
import numpy as np
import df_prepare as dfp
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = dfp.df_from_time_series('Timeseries_PV.csv', beg_year = 2016)


daily_summary = pd.DataFrame()
daily_summary['G_i'] = df.G_i.resample('D').sum()
daily_summary['H_sun'] = df.H_sun.resample('D').sum()
daily_summary['T2m'] = df.T2m.resample('D').mean()
#plt.plot(daily_summary['G_i'])

daily_summary['G_i'].plot(kind='kde')
plt.show()

daily_summary['T2m'].plot(kind='kde')
plt.show()

f, ax = plt.subplots(figsize=(8, 8))

ax = sns.kdeplot(daily_summary['G_i'], daily_summary['T2m'],
                 cmap="Reds", shade=True, shade_lowest=False)
ax.set_aspect(1./ax.get_data_ratio())
plt.grid()
plt.show()


from sklearn.cluster import KMeans

data = pd.DataFrame(daily_summary[['G_i', 'T2m']])

n_clusters = 10

kmeans = KMeans(n_clusters=n_clusters)
X = data[['G_i', 'T2m']]
kmeans.fit(X)
pred = kmeans.predict(X)
data['Cluster'] = pred
#print (f'1: {pred}')
f, ax = plt.subplots(figsize=(16, 8))
sns.scatterplot(data= data, x=list(data['G_i'].index), y='G_i', hue = 'Cluster')
# ax.set_xticklabels(list(data['G_i'].index))

plt.grid()
plt.show()
