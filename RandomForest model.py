import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import scipy.stats as stats

df = pd.read_csv('Data.csv',header=0, index_col=0)

# train-test 80-20 split
from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, 
                                     train_size = 0.8, 
                                     test_size = 0.2, 
                                     random_state = 100)

#RF model without time features
X_train = df_train[['O2 [mg/l]','T [°C]','LF [µS/cm]','pH [--]','Chlorophyll','Abfluss [m³/s]']]
y_train = df_train[['NO3-N [mg/l]']]

X_test = df_test[['O2 [mg/l]','T [°C]','LF [µS/cm]','pH [--]','Chlorophyll','Abfluss [m³/s]']]
y_test = df_test[['NO3-N [mg/l]']]

#RF model performance without time features
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1, max_depth = 10)
scores = cross_validate(rf, X_train, y_train, scoring=['neg_mean_squared_error','r2'], cv=5,n_jobs=-1)
score = scores['test_neg_mean_squared_error'].mean()
r2 = scores['test_r2'].mean()
rmse = np.sqrt(abs(score))
print(rmse, r2)

#RF model with time features                               
X_train = df_train[['O2 [mg/l]','T [°C]','LF [µS/cm]','pH [--]','Chlorophyll','Abfluss [m³/s]','month_sin','month_cos','day_sin','day_cos','hour_sin','hour_cos']]
y_train = df_train[['NO3-N [mg/l]']]

X_test = df_test[['O2 [mg/l]','T [°C]','LF [µS/cm]','pH [--]','Chlorophyll','Abfluss [m³/s]','month_sin','month_cos','day_sin','day_cos','hour_sin','hour_cos']]
y_test = df_test[['NO3-N [mg/l]']]

#RF model performance with time features 
from sklearn.model_selection import cross_validate
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_jobs=-1, max_depth = 10)
scores = cross_validate(rf, X_train, y_train, scoring=['neg_mean_squared_error','r2'], cv=5,n_jobs=-1)
score = scores['test_neg_mean_squared_error'].mean()
r2 = scores['test_r2'].mean()
rmse = np.sqrt(abs(score))
print(rmse, r2)

#Applying RFE to RandomForest model
from sklearn.feature_selection import RFECV
forest = RandomForestRegressor(n_jobs=-1, max_depth = 10)
min_features_to_select = 1  # Minimum number of features to consider
rfecv = RFECV(
    estimator=forest,
    step=1,
    cv=5,
    scoring='neg_mean_squared_error',
    min_features_to_select=min_features_to_select,
)
rfecv.fit(X_train, y_train)

#select features after RFE
features = [f for f,s in zip(X_train.columns, rfecv.support_) if s]
X_train = df_train[features]
y_train = df_train[['NO3-N [mg/l]']]

X_test = df_test[features]
y_test = df_test[['NO3-N [mg/l]']]

#define GridSearch 
from sklearn.model_selection import GridSearchCV
# Create the parameter grid  
param_grid = {
    'bootstrap': [True, False],
    'max_depth': [10, 20, 30],
    'min_samples_leaf': [6, 12, 20],
    'min_samples_split': [6, 12, 20]
}

#Hyperparameter tuning
forest = RandomForestRegressor(n_jobs=-1, max_depth = 10)
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = forest, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)
# Fit the grid search to the data
grid_search.fit(X_train, y_train.values.ravel())
grid_search.best_params_
best_grid = grid_search.best_estimator_

features=X_train.columns[[0,1,2,3,4,5]]
importances = best_grid.feature_importances_
indices = np.argsort(importances)

#print the Feature Importances
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

#RandomForest model performance on unseen data
y_rf = best_grid.predict(X_test)

from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
RMSE_rf=np.sqrt(mean_squared_error(y_test,y_rf))
r2_rf = r2_score(y_test,y_rf)
print(RMSE_rf,r2_rf)
