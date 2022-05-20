import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.stats as stats

df = pd.read_csv('Data.csv',header=0, index_col=0)

#check correlation between targets and variables
X_NAMES1 = ['O2 [mg/l]','T [°C]','LF [µS/cm]','pH [--]','NH4-N [mg/l]','OPO4-P [mg/l]','Chlorophyll','Abfluss [m³/s]']
Y_NAMES1 = ['NO3-N [mg/l]']

corr = df_train.corr()
corr = corr.loc[:, Y_NAMES1]
plt.figure(figsize=(10,10))
sns.heatmap(corr, annot = True)
corr['NO3-N [mg/l]'].sort_values(ascending=False)

# train-test 80-20 split
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, 
                                     train_size = 0.8, 
                                     test_size = 0.2, 
                                     random_state = 100)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_validate

LR = LinearRegression()

#MLR model without time features
X_train = df_train[['O2 [mg/l]','T [°C]','LF [µS/cm]','pH [--]','Chlorophyll','Abfluss [m³/s]']]
y_train = df_train[['NO3-N [mg/l]']]

X_test = df_test[['O2 [mg/l]','T [°C]','LF [µS/cm]','pH [--]','Chlorophyll','Abfluss [m³/s]']]
y_test = df_test[['NO3-N [mg/l]']]

#Train MLR model
scores = cross_validate(LR, X_train, y_train, scoring=['neg_mean_squared_error','r2'], cv=5,n_jobs=-1)
score = scores['test_neg_mean_squared_error'].mean()
r21 = scores['test_r2'].mean()
rmse1 = np.sqrt(abs(score))
print(rmse, r2)

#MLR model with time features
X_train = df_train[['O2 [mg/l]','T [°C]','LF [µS/cm]','pH [--]','Chlorophyll','Abfluss [m³/s]','month_sin','month_cos','day_sin','day_cos','hour_sin','hour_cos']]
y_train = df_train[['NO3-N [mg/l]']]

X_test = df_test[['O2 [mg/l]','T [°C]','LF [µS/cm]','pH [--]','Chlorophyll','Abfluss [m³/s]','month_sin','month_cos','day_sin','day_cos','hour_sin','hour_cos']]
y_test = df_test[['NO3-N [mg/l]']]

#Train MLR model with time features
scores = cross_validate(LR, X_train, y_train, scoring=['neg_mean_squared_error','r2'], cv=5,n_jobs=-1)
score = scores['test_neg_mean_squared_error'].mean()
r2 = scores['test_r2'].mean()
rmse = np.sqrt(abs(score))
print(rmse, r2)

#MLR model performance on unseen data
linear = LinearRegression()
best_LR = linear.fit(X_train,y_train)
y_LR = best_LR.predict(X_test)
RMSE_LR = np.sqrt(mean_squared_error(y_test,y_LR))
r2 = r2_score(y_test,y_LR)
print(RMSE_LR,r2)

