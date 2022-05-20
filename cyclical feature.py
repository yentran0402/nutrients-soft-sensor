import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.stats as stats

df = pd.read_csv('Data.csv',header=0, index_col=0)

#encode time series as cyclical features
def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data

df['month'] = pd.DatetimeIndex(df['Datum']).month
df = encode(df, 'month', 12)

df['day'] = pd.DatetimeIndex(df['Datum']).day
df = encode(df, 'day', 365)

df['hour'] = pd.DatetimeIndex(df['Datum']).hour
df = encode(df, 'hour', 23)

#plot hour without encoding
time = df.iloc[:500]
figure(figsize=(8, 6), dpi=80)
ax = time['hour'].plot()

#plot hour sin
figure(figsize=(8, 6), dpi=80)
ax = time['hour_sin'].plot()

#plot hour cos
figure(figsize=(8, 6), dpi=80)
ax = time['hour_cos'].plot()

#plot sin-cos hour
from matplotlib.pyplot import figure
figure(figsize=(12, 12))
ax = df.plot.scatter('hour_sin', 'hour_cos').set_aspect('equal')

