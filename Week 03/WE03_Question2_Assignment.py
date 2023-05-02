import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

#Load the data

svm_poly_model = pickle.load(open('C:\DSP\WE03\RidMow_assignment_WE05_pickle.csv', "rb"))

#Take inputs from user for an income and lot size
income = float(input("Enter your income: "))
print(income)
lotsize = float(input("Enter your lotsize: "))
print(lotsize)

#find prediction and probability of model
updated_data = pd.DataFrame({'Income': [income], 'Lot_Size' : [lotsize]})
prediction = svm_poly_model.predict(updated_data)
prediction
probability = svm_poly_model.predict_proba(updated_data)
probability

if prediction == 1:
    print('This property is predicted to own a lawnmower') 
else:
    print('This property is predicted NOT to own a lawnmower')