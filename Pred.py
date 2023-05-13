# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 12:54:48 2023

@author: jksls
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


#Data Frames
data     = pd.read_csv('db\Admission_Predict.csv')
new_data = pd.read_csv('db/Admission_Predict_Ver1.1.csv')
new      = new_data.tail(100)

#Used variables
X      = data.drop('Admit', axis=1)
y      = data['Admit']
X_test = new.drop('Admit',axis=1)
y_test = new['Admit']

#Processes
lr     = LinearRegression()
lr.fit(X, y)
y_pred = lr.predict(X_test)
diff   = np.empty(len(y_pred),dtype = float)


for i in range(len(y_pred)):
    diff[i]=y_pred[i]-y_test[i+400]


Predicted_vs_actual = pd.DataFrame({'predicted':y_pred,'actual':y_test,'difference':diff})
print(Predicted_vs_actual)
