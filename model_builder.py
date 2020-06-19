import numpy as np
import pandas as pd
import math
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
data=pd.read_csv('ND2016.csv', sep=',',header=0)
data = np.array(data)
train = data
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

print(len(train))

train=np.array(train)

y_train=np.zeros(len(train))


for i in range(len(train)):
    y_train[i]=train[i][2]
train = np.delete(train, 2, axis=1)

import pickle 

reg = SVR(gamma = 0.0005, kernel = 'rbf', C=500000, verbose=True).fit(train, y_train)

n=10
reg_error = {}

y_pred = reg.predict(train)
for i in range(n):
    error = y_train - y_pred
    reg_error[i] = RandomForestRegressor(max_depth = 30, n_estimators = 500).fit(train, error)
    error_pred = reg_error[i].predict(train)
    y_pred = y_pred + error_pred

# save all 10 models to joblib

pickle.dump(reg, open('base_model.sav', 'wb'))
pickle.dump(reg_error[0], open('error_model_1.sav','wb'))
pickle.dump(reg_error[1], open('error_model_2.sav','wb'))
pickle.dump(reg_error[2], open('error_model_3.sav','wb'))
pickle.dump(reg_error[3], open('error_model_4.sav','wb'))
pickle.dump(reg_error[4], open('error_model_5.sav','wb'))
pickle.dump(reg_error[5], open('error_model_6.sav','wb'))
pickle.dump(reg_error[6], open('error_model_7.sav','wb'))
pickle.dump(reg_error[7], open('error_model_8.sav','wb'))
pickle.dump(reg_error[8], open('error_model_9.sav','wb'))
pickle.dump(reg_error[9], open('error_model_10.sav','wb'))
print("MODELS BUILT AND SAVED")



