# -*- coding: utf-8 -*-
"""
Created on Wed May 12 14:39:33 2021

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

from statsmodels.tsa.arima.model import ARIMA  #ARIMA  model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
from pmdarima.arima import auto_arima
from math import sqrt
from sklearn.metrics import mean_squared_error

bitcoin = pd.read_csv(r"E:\PhD study\ELEG5491 Introduction to Deep Learning\bitcoin\datasets\bitcoin1dim.csv")
bitcoin1 = bitcoin.groupby([pd.Grouper(key='Date')]).first().reset_index()
df_check=bitcoin.isnull().values.any()
print(df_check)
bitcoin1 = bitcoin1.set_index('Date')

bitcoin1.plot()
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)

bitcoin_diff1 = bitcoin1.diff(1)
bitcoin_diff1 = bitcoin_diff1.dropna()
bitcoin_diff1.plot(style='', figsize=(15,5), label='first order difference')
plt.xticks(fontproperties = 'Times New Roman', size = 14)
plt.yticks(fontproperties = 'Times New Roman', size = 14)
plt.title('first order difference')

print(sm.tsa.stattools.adfuller(bitcoin_diff1))

#Draw the picture of ACF
acf = plot_acf(bitcoin_diff1, lags=30)
plt.title('ACF')
acf.show()
#Draw the picture of PACF
pacf = plot_pacf(bitcoin_diff1, lags=30)
plt.title('PACF')
pacf.show()
plt.show()


splitdate = '2019-12-17'
bitcoin_train1 = bitcoin.loc[bitcoin.Date <= splitdate]
print(bitcoin_train1)
bitcoin_test1 = bitcoin.loc[bitcoin.Date > splitdate]
print(bitcoin_test1)

bitcoin_train2 = bitcoin_train1.drop(['Date'],axis=1)
bitcoin_test2 = bitcoin_test1.drop(['Date'],axis=1)

model1 = ARIMA(bitcoin_train2, order=(3,1,3))
results = model1.fit()
print(results.summary())
pred2 = results.predict(1885,2355, dynamic=True, typ='levels')
print(pred2)
pred2.plot()

rmse = sqrt(mean_squared_error(pred2, bitcoin_test2))
print('Test RMSE: %.3f' % rmse)