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


import sys


from itertools import product
variablesPosibles=[
[25500, 3000,34500],
[round(40*0.85,2), 40, round(40*1.15,2)],
[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46],
[44115, 51900, 59685],
[round(25*0.85,2), 25, round(25*1.15,2)],
[0.013, 0.016, 0.018],
[0.011, 0.013, 0.015],
[0.3672, 0.432, 0.4968],
[0.17, 0.2,0.23],
[0.017, 0.02, 0.023],
[148, 173, 199],
[1,2,3]
]
flujo_masico_aceite=[51900*0.85,51900,51900*1.15]
temperatura_entrada_aceite=[40*0.85, 40, 40*1.15]
temperatura_salida_aceite=[21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46]
flujo_masico_agua=[30000*0.85, 30000, 30000*1.15]
temperatura_entrada_agua=[25*0.85, 25, 25*1.15]
diametro_tubos_exterior=[0.013, 0.016, 0.018]
diametro_tubos_interior=[0.011, 0.013, 0.015]
diametro_carcasa_interior=[432*0.85, 432, 432*1.15]
espacio_baffles=[0.2*0.85, 0.2, 0.2*1.15]
pitch=[0.02*0.85, 0.02, 0.02*1.15]
numero_tubos=[148, 173, 199]
numero_pasos=[1, 2, 3]

flujo_masico_aceite_Cliente0=30000
temperatura_entrada_aceite_Cliente1=40
flujo_masico_agua_Cliente3=51900
temperatura_entrada_agua_Cliente4=25


def cartesian(arrays, out=None):
    """
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    print(dtype)

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=np.float64)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], int(m))
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:int(m),1:])
        for j in range(1, arrays[0].size):
            out[j*int(m):(j+1)*int(m),1:] = out[0:int(m),1:]
    return out

def find_nearest(array,valor):
	mayor=sys.maxsize
	menor=0
	idxMayor=0
	idxMenor=0
	for i in range(0, len(array)):
		if (array[i][12]>menor and array[i][12]<=(valor*1)):
			menor=array[i][12]
			idxMenor=i
		if (array[i][12]<mayor and array[i][12]>=(valor*1)):
			mayor=array[i][12]
			idxMayor=i
	return idxMenor, idxMayor



def returnValues(data, indexMenor,indexMayor):

	rowMenor=data[indexMenor,0:12]
	rowMayor=data[indexMayor,0:12]
	return rowMenor, rowMayor

def generarEntradasGeneticas(rowMenor, rowMayor):
	arrayValidacion=[]

	for i in range(0, len(rowMenor)):
		 # and i!=10
		if(i!=0 and i!=1 and i!=3 and i!=4):
			if(rowMenor[i]==rowMayor[i]):
				arrayNuevo=[]
				arrayNuevo.append(rowMenor[i])
				
				arrayValidacion.append(arrayNuevo)
			else:
				arrayValidacion.append(variablesPosibles[i])
		else:
			if i==0 and flujo_masico_aceite_Cliente0!=-1:
				arrayNuevo=[]
				arrayNuevo.append(flujo_masico_aceite_Cliente0)
				arrayValidacion.append(arrayNuevo)
			else:
				if i==1 and temperatura_entrada_aceite_Cliente1!=-1:
					arrayNuevo=[]
					arrayNuevo.append(temperatura_entrada_aceite_Cliente1)
					arrayValidacion.append(arrayNuevo)
				else:
					if i==3 and flujo_masico_agua_Cliente3!=-1:
						arrayNuevo=[]
						arrayNuevo.append(flujo_masico_agua_Cliente3)
						arrayValidacion.append(arrayNuevo)
					else:
						if i==4 and temperatura_entrada_agua_Cliente4!=-1:
							arrayNuevo=[]
							arrayNuevo.append(temperatura_entrada_agua_Cliente4)
							arrayValidacion.append(arrayNuevo)
						# else:
						# 	if i==10 :
						# 		arrayNuevo=[]
						# 		arrayNuevo.append(173)
						# 		arrayValidacion.append(arrayNuevo)
	cartesian2= cartesian(arrayValidacion)
	return cartesian2

def get_model(n_inputs, n_outputs):
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(13626, activation='relu'))

	model.add(Dense(n_outputs, activation="linear"))
	model.compile(loss='mse', optimizer='adam',metrics=['mse','mae','accuracy'])
	return model

def predict_inverse(valorReferencia, training_data_not_scaled, model):
	# valorReferencia=77583
	indexMenor, indexMayor=find_nearest(training_data_not_scaled,valorReferencia)

	rowDown, rowUp= returnValues(training_data_not_scaled,indexMenor,indexMayor)
	geneticas = generarEntradasGeneticas(rowDown,rowUp)

	arrayValidas=[]
	for linea in geneticas:
		
		newX = asarray([linea])
		geneticas_scaled = scaler_x.transform (newX)

		yhat=model.predict(geneticas_scaled)
		yInverse=scaler_y.inverse_transform(yhat)
		if(yInverse[0][0] >(valorReferencia*0.95)and yInverse[0][0]<(valorReferencia*1.05)and yInverse[0][1]>0 and yInverse[0][1]<1 and yInverse[0][2] >0 and yInverse[0][2] <1):
			
			geneticaSinEscalar=newX
			listaGenetica=(geneticaSinEscalar[0]).tolist()
			listaGenetica.append(yInverse[0][0])
			listaGenetica.append(yInverse[0][1] )
			listaGenetica.append(yInverse[0][2] )
			arrayValidas.append(listaGenetica)
	return arrayValidas

def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	print("Inputs: "+str(n_inputs))
	print("Outputs: "+str(n_outputs))
	
	
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
	
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
	
	history = model.fit(xtrain_scale, ytrain_scale, verbose=1, epochs=100, validation_data=(xval_scale,yval_scale))
	
	
	# row=[30000, 40, 34, 51900, 25, 0.016, 0.013, 0.432, 0.17, 0.02, 173, 2]

	# newX = asarray([row])
	# scalada= scaler_x.transform(newX)
	# yhat=model.predict(scalada)
	# print(yhat)
	# yInverse=scaler_y.inverse_transform(yhat)
	# print(yInverse)
	return model



training_data_not_scaled = genfromtxt('datosUnidosTodos.csv', delimiter=';', skip_header=0)
X = training_data_not_scaled[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
y=training_data_not_scaled[:,[12,13,14]]


print(X)
print("------------------------")
print(y)

#evaluate model
modelo = evaluate_model(X, y)
#Calor a predecir
valorReferencia=77583
arrayValidas=predict_inverse(valorReferencia,training_data_not_scaled,modelo)
print(arrayValidas)
