from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostRegressor
import pandas as pd
import numpy as np

# Load Data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')


# Hyperparameters
param_grid = {
              'n_estimators': np.arange(450, 490, 5),
              'learning_rate': np.arange(3, 4, 0.1)
             }

# GridSearchCV
gs = GridSearchCV(estimator=AdaBoostRegressor(random_state=12),
                  param_grid=param_grid,
                  scoring=['r2', 'neg_root_mean_squared_error'],
                  refit='r2',
                  cv=10,
                  n_jobs=-1,
                  verbose=4)

gs = gs.fit(X_train, y_train.values.ravel())