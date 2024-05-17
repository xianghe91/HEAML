from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

# Load Data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')


# Hyperparameters
param_grid = {
              "n_estimators": [20, 50, 100, 200],
              "max_depth": [10, 15, 20],
              "min_samples_leaf": [5, 10, 20],
              "min_samples_split": [2, 5, 10]  
             }

# GridSearchCV
gs = GridSearchCV(estimator=GradientBoostingRegressor(random_state=12),
                  param_grid=param_grid,
                  scoring=['r2', 'neg_root_mean_squared_error'],
                  refit='r2',
                  cv=10,
                  n_jobs=-1,
                  verbose=4)

gs = gs.fit(X_train, y_train.values.ravel())