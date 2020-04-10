import numpy as np
import pandas as pd
import pickle

# load models
base_model = pickle.load(open('base_model.sav','rb'))  
error_model_1 = pickle.load(open('error_model_1.sav','rb'))
error_model_2 = pickle.load(open('error_model_2.sav','rb'))
error_model_3 = pickle.load(open('error_model_3.sav','rb'))
error_model_4 = pickle.load(open('error_model_4.sav','rb'))
error_model_5 = pickle.load(open('error_model_5.sav','rb'))
error_model_6 = pickle.load(open('error_model_6.sav','rb'))
error_model_7 = pickle.load(open('error_model_7.sav','rb'))
error_model_8 = pickle.load(open('error_model_8.sav','rb'))
error_model_9 = pickle.load(open('error_model_9.sav','rb'))
error_model_10 = pickle.load(open('error_model_10.sav','rb'))

print ("INSTRUCTIONS : \n - Enter the exact filename with .csv extension \n - Make sure the file is present in the same folder as the driver script \n - Make sure file does not contain header row \n - File must only contain 2 columns. First column should contain Z values. Second column should contain N values \n")
filename = input("Enter file name: ")
data=pd.read_csv(str(filename), sep=',',header=None, usecols=[0,1])
data = np.array(data)
res = base_model.predict(data)
err1 = error_model_1.predict(data)
err2 = error_model_2.predict(data)
err3 = error_model_3.predict(data)
err4 = error_model_4.predict(data)
err5 = error_model_5.predict(data)
err6 = error_model_6.predict(data)
err7 = error_model_7.predict(data)
err8 = error_model_8.predict(data)
err9 = error_model_9.predict(data)
err10 = error_model_10.predict(data)
res = res+err1+err2+err3+err4+err5+err6+err7+err8+err9+err10

dash = '-' * 40
print(dash)
print('{:<10s}{:>4s}{:>20s}'.format("Z","N","BE (MeV)"))
print(dash)

for i in range(len(data)):
    print('{:<10d}{:>4d}{:>20f}'.format(data[i][0],data[i][1], res[i]) )
