# use mlp for prediction on multi-output regression
from numpy import asarray
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense
import tensorflow as tf
from numpy import genfromtxt
from keras.layers import LSTM,Dense ,Dropout

from random import random
from numpy import array
from numpy import cumsum
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold


# get the dataset
def get_dataset():
	X, y = make_regression(n_samples=146707, n_features=12, n_informative=5, n_targets=3, random_state=2)
	return X, y
 
# # get the model
# def get_model(n_inputs, n_outputs):
# 	model = Sequential()
# 	model.add(Dense(12, input_dim=n_inputs, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(13626,activation='relu'))
# 	model.add(Dense(n_outputs, activation='linear'))
# 	model.compile(loss='mae', optimizer='adam')
# 	return model

# # get the model
# def get_model(n_inputs, n_outputs):
# 	model = Sequential()
# 	model.add(Dense(12, input_dim=n_inputs, kernel_initializer='normal', activation='relu'))
# 	model.add(Dense(13626,activation='relu'))
# 	model.add(Dense(n_outputs, activation='linear'))
# 	model.compile(loss='mae', optimizer='adam')
# 	return model

def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(13626, activation='relu'))

	model.add(Dense(n_outputs, activation="linear"))
	model.compile(loss='mse', optimizer='adam',metrics=['mse','mae','accuracy'])
	return model

def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	print("Inputs: "+str(n_inputs))
	print("Outputs: "+str(n_outputs))
	
	# define evaluation procedure
	# cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# # enumerate folds
	# train_ix, test_ix=	cv.split(X)[0]
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
	
	# train_ix, test_ix=train_test_split(X, shuffle=False)
	# # for train_ix, test_ix in cv.split(X):
	# # prepare data
	# X_train, X_val = X[train_ix], X[test_ix]
	# y_train, y_val = y[train_ix], y[test_ix]
	# # define model
	print("YYYYYYYYYYYYYYY")
	print(y_train)
	print("------------")
	print(y_val)

	# y_train = np.reshape (y_train, (-1,1)) 
	# y_val = np.reshape (y_val, (-1,1))
	scaler_x = MinMaxScaler () 
	scaler_y = MinMaxScaler ()
	print (scaler_x.fit (X_train)) 
	xtrain_scale = scaler_x.transform (X_train) 
	print (scaler_x.fit (X_val)) 
	xval_scale = scaler_x.transform (X_val)
	print (scaler_y.fit (y_train)) 
	ytrain_scale = scaler_y.transform (y_train) 
	print (scaler_y.fit (y_val)) 
	yval_scale = scaler_y.transform (y_val)




	model = get_model(n_inputs, n_outputs)
	# fit model
	# print(xtrain_scale)
	print(xtrain_scale)
	print("......................")
	print(ytrain_scale)
	# model.compile(loss='mse',
#             optimizer='adam',
#             metrics=['mse','mae','accuracy'])
	history = model.fit(xtrain_scale, ytrain_scale, verbose=1, epochs=100, validation_data=(xval_scale,yval_scale))
	# evaluate model on test set
	mae = model.evaluate(xval_scale, yval_scale, verbose=1)



	print(history.history.keys())
# "Loss"
	fig, axs = plt.subplots(2)
	fig.suptitle('Heat Exchange and Pressure drop on tube side')
	axs[0].plot(history.history['loss'])
	axs[0].plot(history.history['val_loss'])
	# plt.plot(history.history['val_loss'])
	# plt.plot(history.history['acc'])
	axs[0].set_title('Loss')
	axs[0].set_ylabel('loss')
	axs[0].set_xlabel('epoch')
	axs[0].legend(['train', 'validation'], bbox_to_anchor=(1.05, 1), loc='upper left')
	# plt.show()


	print(history.history.keys())
# "Loss"
	# plt.plot(history.history['loss'])
	# plt.plot(history.history['val_loss'])
	axs[1].plot(history.history['acc'])
	axs[1].plot(history.history['val_acc'])
	axs[1].set_title('Accuracy')
	axs[1].set_ylabel('acc')
	axs[1].set_xlabel('epoch')
	axs[1].legend(['train', 'validation'], bbox_to_anchor=(1.05, 1), loc='upper left')
	fig.tight_layout(pad=3.0)
	fig.savefig('Nuevas/Finales/tubocalor',  bbox_inches='tight')

	plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.1)
	plt.show()
	# store result
	# print('>%.3f' % str(mae))
	#results.append(mae)
	return results


# def evaluate_model(X, y):
# 	results = list()
# 	n_inputs, n_outputs = X.shape[1], y.shape[1]
# 	print("Inputs: "+str(n_inputs))
# 	print("Outputs: "+str(n_outputs))
	
# 	# define evaluation procedure
# 	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
# 	# enumerate folds
# 	for train_ix, test_ix in cv.split(X):
# 		# prepare data
# 		X_train, X_test = X[train_ix], X[test_ix]
# 		y_train, y_test = y[train_ix], y[test_ix]
# 		# define model
# 		model = get_model(n_inputs, n_outputs)
# 		# fit model
# 		model.fit(X_train, y_train, verbose=1, epochs=100)
# 		# evaluate model on test set
# 		mae = model.evaluate(X_test, y_test, verbose=1)
# 		# store result
# 		print('>%.3f' % mae)
# 		results.append(mae)
# 	return results


# load dataset

# training_data_not_scaled = genfromtxt('datosUnidosCalor.csv', delimiter=';', skip_header=0)
# X = training_data_not_scaled[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
# y=training_data_not_scaled[:,[12]]


training_data_not_scaled = genfromtxt('datosUnidosTodos.csv', delimiter=';', skip_header=0)
X = training_data_not_scaled[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
y=training_data_not_scaled[:,[12,13]]


# X, y = get_dataset()
print(X)
print("------------------------")
print(y)

#evaluate model
results = evaluate_model(X, y)
# summarize performance
print('MAE: %.3f (%.3f)' % (mean(results), std(results)))

# n_inputs, n_outputs = X.shape[1], y.shape[1]
# # get model
# model = get_model(n_inputs, n_outputs)
# # fit the model on all data
# model.fit(X, y, verbose=0, epochs=100)
# # make a prediction for new data
# row = [-0.99859353,2.19284309,-0.42632569,-0.21043258,-1.13655612,-0.55671602,-0.63169045,-0.87625098,-0.99445578,-0.3677487]
# newX = asarray([row])
# yhat = model.predict(newX)
# print('Predicted: %s' % yhat[0])