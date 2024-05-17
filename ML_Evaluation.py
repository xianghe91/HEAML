from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np


# Load Data
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
y_train = y_train.values.ravel()

X_test = pd.read_csv('X_test.csv')
y_test = pd.read_csv('y_test.csv')
y_test = y_test.values.ravel()


# Models
RF_model = RandomForestRegressor(random_state=12, max_depth=15, min_samples_leaf=5, min_samples_split=2, n_estimators=200)
GBR_model = GradientBoostingRegressor(random_state=12, max_depth=15, min_samples_leaf=20, min_samples_split=2, n_estimators=200)
ABR_model = AdaBoostRegressor(random_state=12, learning_rate=3.2, n_estimators=460)
ETR_model = ExtraTreesRegressor(random_state=12, max_depth=15, min_samples_leaf=5, min_samples_split=2, n_estimators=200)

models = [RF_model, GBR_model, ABR_model, ETR_model]


# Evaluating Models
for i in np.arange(0, len(models)):
    model = models[i]
    model.fit(X_train, y_train)
    y_train_predict = model.predict(X_train)
    r2_train = r2_score(y_train, y_train_predict)
    rmse_train = mean_squared_error(y_train, y_train_predict, squared=False)

    y_test_predict = model.predict(X_test)
    r2_test = r2_score(y_test, y_test_predict)
    rmse_test = mean_squared_error(y_test, y_test_predict, squared=False)