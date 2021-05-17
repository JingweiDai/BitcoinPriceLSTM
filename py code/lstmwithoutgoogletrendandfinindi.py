# -*- coding: utf-8 -*-
"""
Created on Sun May 16 11:33:29 2021

@author: Jingwei Dai
"""

import pandas as pd
import numpy as np
import datetime
import pytz  #function of time region
import statsmodels.api as sm  # Unit root test
import matplotlib.pyplot as plt 
import seaborn as sns
import matplotlib.dates as mdate

from keras.layers import Dropout
from statsmodels.tsa.arima.model import ARIMA  #ARIMA  model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error

bitcoin = pd.read_csv(r"E:\PhD study\ELEG5491 Introduction to Deep Learning\bitcoin\datasets\bitcoin1dimtrain.csv")
training_set1=bitcoin.values            #converting to 2d array
training_set1 

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler()                           #scaling using normalisation 
training_set1 = sc.fit_transform(training_set1)
xtrain=training_set1[0:1884]                  #input values of rows [0-2694]		   
ytrain=training_set1[1:1885] 

today=pd.DataFrame(xtrain)               #assigning the values of xtrain to today
tomorrow=pd.DataFrame(ytrain)            #assigning the values of xtrain to tomorrow
ex= pd.concat([today,tomorrow],axis=1)        #concat two columns 
ex.columns=(['today','tomorrow'])
xtrain = np.reshape(xtrain, (1884, 1, 1))

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

regressor=Sequential()                                                      #initialize the RNN
regressor.add(LSTM(units=4,activation='sigmoid',input_shape=(None,1))) 
regressor.add(Dropout(0.01))     #adding input layerand the LSTM layer 
regressor.add(Dense(units=1))                                               #adding output layers
regressor.compile(optimizer='adam',loss='mean_squared_error')               #compiling the RNN
regressor.fit(xtrain,ytrain,batch_size=25,epochs=30)   

test_set = pd.read_csv(r"E:\PhD study\ELEG5491 Introduction to Deep Learning\bitcoin\datasets\bitcoin1dimtest.csv")
test_set.head()

inputs = test_set.values      #converting to 2D array		
inputs = sc.fit_transform(inputs)
inputs = np.reshape(inputs, (471, 1, 1))
predicted_price = regressor.predict(inputs)
predicted_price = sc.inverse_transform(predicted_price)
print(predicted_price)
testScore = sqrt(mean_squared_error(predicted_price, test_set))
print('Test Score: %.2f RMSE' % (testScore))
