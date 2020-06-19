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
train, test1 = train_test_split(data, test_size=0.4, shuffle=True, random_state = 1)
test, val = train_test_split(test1, test_size=0.5, shuffle=True, random_state = 1)

from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
print(len(train))
print(len(test))
print(len(val))
import matplotlib as mpl
mpl.rcParams['xtick.major.size'] = 4
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['xtick.minor.size'] = 2
mpl.rcParams['xtick.minor.width'] = 2
mpl.rcParams['ytick.major.size'] = 2
mpl.rcParams['ytick.major.width'] = 2
mpl.rcParams['ytick.minor.size'] = 2
mpl.rcParams['ytick.minor.width'] = 2
plt.rcParams['xtick.labelsize'] = 20
plt.rcParams['ytick.labelsize'] = 20


train=np.array(train)
test=np.array(test)
val=np.array(val)
y_train=np.zeros(len(train))
y_test=np.zeros(len(test))
y_val=np.zeros(len(val))


print(len(test))
for i in range(len(train)):
    y_train[i]=train[i][2]
train = np.delete(train, 2, axis=1)
for i in range(len(test)):
    y_test[i]=test[i][2]
test = np.delete(test, 2, axis=1)
for i in range(len(val)):
    y_val[i]=val[i][2]
val = np.delete(val, 2, axis=1)

reg = SVR(gamma = 0.0005, kernel = 'rbf', C=500000, verbose=True).fit(train, y_train)
n=15

reg_error = {}

N_train = train[:,1]
N_test = test[:,1]
N_val = val[:,1]

y_pred = reg.predict(train)
for i in range(n):
    error = y_train - y_pred
    reg_error[i] = RandomForestRegressor(max_depth = 30, n_estimators = 500).fit(train, error)
    error_pred = reg_error[i].predict(train)
    y_pred = y_pred + error_pred

#calculate number of iterations using val set

y_pred_val = reg.predict(val)
error = mean_squared_error(y_val, y_pred_val)
print("RMS Error of BASE MODEL (val): %f "% math.sqrt(error))

rms=[]
from matplotlib.ticker import StrMethodFormatter
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.02f}')) # 2 decimal places

error = y_val - y_pred_val
print("RMS Error: %f for BASE MODEL "% (math.sqrt(mean_squared_error(y_val, y_pred_val))))
legend_properties = {'weight':'bold', 'size':16}
plt.xlabel('Neutron Number (N)', fontsize=24, fontweight='bold', labelpad=3)
plt.ylabel("Error (MeV)", fontsize=24, fontweight='bold', labelpad=3)
plt.scatter(N_val,error, label='SVM')
plt.legend(loc='upper right', prop = legend_properties)
plt.show()
for i in range(n):
    error_pred_val = reg_error[i].predict(val)
    y_pred_val = y_pred_val + error_pred_val
    error = y_val - y_pred_val
    print("RMS Error: %f for iteration no. %d "% (math.sqrt(mean_squared_error(y_val, y_pred_val)), (i+1)))
    rms = np.append(rms, math.sqrt(mean_squared_error(y_val, y_pred_val)))
print(rms) 

legend_properties = {'weight':'bold', 'size':16}
plt.plot(np.arange(1, 16, 1.0), rms, label='SVM')
plt.xticks(np.arange(1, 16, 1.0))
plt.yticks(np.arange(min(rms), max(rms), 0.05))
plt.locator_params(axis='y', nbins=4)
plt.xlabel('iteration', size=24, fontweight='bold')
plt.ylabel('RMSE (MeV)', size=24, fontweight='bold')
plt.legend(loc='upper right', prop = legend_properties)
plt.gca().yaxis.set_major_formatter(StrMethodFormatter('{x:,.02f}')) # 2 decimal places
plt.show()

#make final prediction

y_pred_test = reg.predict(test)
error = mean_squared_error(y_test, y_pred_test)
print("RMS Error of BASE MODEL (test): %f "% math.sqrt(error))

error = y_test - y_pred_test
for i in range(10):
    error = y_test - y_pred_test
    error_pred_test = reg_error[i].predict(test)
    y_pred_test = y_pred_test + error_pred_test
print("Final RMS Error: %f"% (math.sqrt(mean_squared_error(y_test, y_pred_test))))
plt.scatter(N_test,error)
plt.show()


