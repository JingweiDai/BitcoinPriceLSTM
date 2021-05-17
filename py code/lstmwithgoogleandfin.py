# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:22:09 2021

@author: Jingwei Dai
"""
import pandas as pd 
from pandas import DataFrame
from pandas import concat
from math import sqrt
from numpy import concatenate
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as pyplot 
import numpy as np 
from sklearn.preprocessing import MinMaxScaler
from keras import Sequential
from keras.layers import LSTM, Dense, Dropout, Activation
from pandas import read_csv

# Load dataset by using Pandas library 
dataset = pd.read_csv(r"E:\PhD study\ELEG5491 Introduction to Deep Learning\bitcoin\datasets\bitcoinwithgoogleandfin.csv", header=0, index_col=0)
values = dataset.values
print(dataset)
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# Here is created input columns which are (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# Here is created output/forecast column which are (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# Dataset values are normalized by using MinMax method
scaler = MinMaxScaler(feature_range=(0,1))
scaled = scaler.fit_transform(values)
print(len(scaled))

# Normalized values are converted for supervised learning 
reframed = series_to_supervised(scaled,1,1,True)

# Dataset is splitted into two groups which are train and test sets
values = reframed.values 
train_size = int(len(values)*0.60)
validation_size = int(len(values)*0.80)
train = values[:train_size,:]
validation =values[(train_size+1):validation_size,:]
test = values[validation_size:,:]
print(train)
print(test)

# Splitted datasets are splitted to trainX, trainY, testX and testY 
trainX, trainY = train[:,:-1], train[:,13]
validationX, validationY = validation[:,:-1], validation[:,13]
testX, testY = test[:,:-1], test[:,13]
print(trainY, trainY.shape)

# Train and Test datasets are reshaped to be used in LSTM
trainX = trainX.reshape((trainX.shape[0],1,trainX.shape[1]))
validationX = validationX.reshape((validationX.shape[0],1,validationX.shape[1]))
testX = testX.reshape((testX.shape[0],1,testX.shape[1]))
print(trainX.shape, trainY.shape,testX.shape,testY.shape)

# LSTM model is created and adjusted neuron structure
model = Sequential()
model.add(LSTM(128, input_shape=(trainX.shape[1], trainX.shape[2])))
model.add(Dropout(0.01))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# Dataset is trained by using trainX and trainY, validated by validationX and validationY
history = model.fit(trainX, trainY, epochs=30, batch_size=25, validation_data=(validationX, validationY), verbose=2, shuffle=False)

# Loss values are calculated for every training epoch and are visualized
pyplot.plot(history.history['loss'], label='train')
pyplot.plot(history.history['val_loss'], label='validation')
pyplot.title("Train and Validation set Loss Value Rate")
pyplot.xlabel('Epochs Number', fontsize=12)
pyplot.ylabel('Loss Value', fontsize=12)
pyplot.legend()
pyplot.show()

# Prediction process is performed for train dataset
trainPredict = model.predict(trainX)
trainX = trainX.reshape((trainX.shape[0], trainX.shape[2]))

# Prediction process is performed for validation dataset
validationPredict = model.predict(validationX)
validationX = validationX.reshape((validationX.shape[0], validationX.shape[2]))

# Prediction process is performed for test dataset
testPredict = model.predict(testX)
testX = testX.reshape((testX.shape[0], testX.shape[2]))

# Train dataset inverts scaling for training
trainPredict = concatenate((trainPredict, trainX[:, -9:]), axis=1)
trainPredict = scaler.inverse_transform(trainPredict)
trainPredict = trainPredict[:,0]

# Validation dataset inverts scaling for training
validationPredict = concatenate((validationPredict, validationX[:, -9:]), axis=1)
validationPredict = scaler.inverse_transform(validationPredict)
validationPredict = validationPredict[:,0]

# Test dataset inverts scaling for forecasting
testPredict = concatenate((testPredict, testX[:, -9:]), axis=1)
testPredict = scaler.inverse_transform(testPredict)
testPredict = testPredict[:,0]

# invert scaling for actual
trainY = trainY.reshape((len(trainY), 1))
inv_trainy = concatenate((trainY, trainX[:, -9:]), axis=1)
inv_trainy = scaler.inverse_transform(inv_trainy)
inv_trainy = inv_trainy[:,0]

validationY = validationY.reshape((len(validationY), 1))
inv_validationy = concatenate((validationY, validationX[:, -9:]), axis=1)
inv_validationy = scaler.inverse_transform(inv_validationy)
inv_validationy = inv_validationy[:,0]

testY = testY.reshape((len(testY), 1))
inv_testy = concatenate((testY, testX[:, -9:]), axis=1)
inv_testy = scaler.inverse_transform(inv_testy)
inv_testy = inv_testy[:,0]

#It should be noted that RMSE would be different each time run the code
#becasue of dropout layer.
# Performance measure calculated by using mean_squared_error for train and test prediction
rmset = sqrt(mean_squared_error(inv_trainy, trainPredict))
print('Train RMSE: %.3f' % rmset)
rmsev = sqrt(mean_squared_error(inv_validationy, validationPredict))
print('Validation RMSE: %.3f' % rmsev)
rmse = sqrt(mean_squared_error(inv_testy, testPredict))
print('Test RMSE: %.3f' % rmse)

# Three parts of datasets are concatenated
final = np.append(trainPredict, validationPredict)
final = np.append(final, testPredict)
final = pd.DataFrame(data=final, columns=['Close'])
actual = dataset.Close
actual = actual.values
actual = pd.DataFrame(data=actual, columns=['Close'])

# Finally result are visualized
pyplot.plot(actual.Close, 'b', label='Real Value')
pyplot.plot(final.Close[1884:len(final)], 'g' , label='Predicted Value')
pyplot.title("Daily Bitcoin Predicted Prices")
pyplot.xlabel('Daily Time', fontsize=12)
pyplot.ylabel('Close Price', fontsize=12)
pyplot.legend(loc='best')
pyplot.show()