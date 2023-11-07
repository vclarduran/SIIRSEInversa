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

import pandas as pd
import os
import sys
from itertools import product

datos = "parcial.csv" #CAMBIAR EL CSV
columnasAUsar = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
variablesAPredecir = 1
columnaPredecida = 4
#COLUMNA 4 = Power



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


def find_nearest(df, valor, column_number):
    print(df)
    mayor = sys.maxsize
    menor = 0
    idxMayor = 0
    idxMenor = 0

    for i in range(len(df)):
        if (df.iloc[i, column_number] > menor and df.iloc[i, column_number] <= (valor * 1)):
            menor = df.iloc[i, column_number]
            idxMenor = i
        if (df.iloc[i, column_number] < mayor and df.iloc[i, column_number] >= (valor * 1)):
            mayor = df.iloc[i, column_number]
            idxMayor = i

    return idxMenor, idxMayor


def returnValues(data, indexMenor,indexMayor):

	rowMenor=data.iloc[indexMenor,0:12]
	rowMayor=data.iloc[indexMayor,0:12]
	return rowMenor, rowMayor

def generarEntradasGeneticas(rowMenor, rowMayor, variablesPosibles):
	arrayValidacion=[]

	for columna in columnasAUsar:
		arrayColumna =[]
		if(rowMayor.iloc[columna] == rowMenor.iloc[columna]):
			arrayColumna.append(rowMayor.iloc[columna])
			arrayValidacion.append(arrayColumna)
		else:
			if(columna>5):
				arrayValidacion.append(valoresPosibles.iloc[columna-2])
			else:
				arrayValidacion.append(valoresPosibles.iloc[columna-1])
	
	cartesian2= cartesian(arrayValidacion)
	return cartesian2

def get_model(n_inputs, n_outputs): #Crea un modelo de dos capas 
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(13626, activation='relu'))

	model.add(Dense(n_outputs, activation="linear"))
	model.compile(loss='mse', optimizer='adam',metrics=['mse','mae','accuracy'])
	return model


def predict_inverse(valorReferencia, training_data_not_scaled, model, posiblesConfiguraciones): #Extrae todas las configuraciones posibles, si está dentro del rango la añade al array validado
	# valorReferencia=77583
	indexMenor, indexMayor=find_nearest(training_data_not_scaled,valorReferencia, columnaPredecida)

	rowDown, rowUp= returnValues(training_data_not_scaled,indexMenor,indexMayor)
	geneticas = generarEntradasGeneticas(rowDown,rowUp, posiblesConfiguraciones)

	scaler_x = MinMaxScaler () 
	scaler_y = MinMaxScaler ()

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

def evaluate_model(X, y): #Crea el modelo necesario con los datos que tenemos
	results = list()
	n_inputs, n_outputs = len(columnasAUsar), variablesAPredecir
	print("Inputs: "+str(n_inputs))
	print("Outputs: "+str(n_outputs))
	
	
	X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42)
	
	scaler_x = MinMaxScaler () 
	scaler_y = MinMaxScaler ()
	print (scaler_x.fit (X_train)) 
	xtrain_scale = scaler_x.transform (X_train) 
	print (scaler_x.fit (X_val)) 
	xval_scale = scaler_x.transform (X_val)
	y_train = y_train.values.reshape(-1, 1)
	y_val = y_val.values.reshape(-1, 1) 
	print(scaler_y.fit(y_train))
	ytrain_scale = scaler_y.transform(y_train)
	print(scaler_y.fit(y_val))
	yval_scale = scaler_y.transform(y_val)


	model = get_model(n_inputs, n_outputs)
	
	#CAMBIAR EPOCHS
	history = model.fit(xtrain_scale, ytrain_scale, verbose=1, epochs=10, validation_data=(xval_scale,yval_scale))
	
	return model

def valoresPosibles(datos):
	#lectura csv
	df = pd.read_csv(os.path.join(datos), delimiter=",")

	#buscar min y max
	minimos = df.min()
	maximos = df.max()

	#para cada columna

	variablesPosibles=[]

	for columna in columnasAUsar:
		min = round((minimos.iloc[columna]*0.95),12)
		max = round((maximos.iloc[columna]*1.05),12) 
		valor = min
		valoresDeVariable = []
		valoresDeVariable.append(valor) 
		while(valor < max):
			valoresDeVariable.append(round(valor,12))
			valor+=0.1
		variablesPosibles.append(valoresDeVariable)    
	return variablesPosibles


#EMPIEZA EL MAIN
posiblesConfiguraciones = valoresPosibles(datos)

dtype = [
    #('Timestamp', 'datetime64'),
    ('Wind speed (m/s)', 'float64'),
    ('Wind direction (°)', 'float64'),
    ('Nacelle position (°)', 'float64'),
    ('Energy Export (kWh)', 'float64'),
    ('Power (kW)', 'float64'),
    ('Rotor current (A)', 'float64'),
    ('Rotor speed (RPM)', 'float64'),
    ('Generator RPM (RPM)', 'float64'),
    ('Generator RPM, Max (RPM)', 'float64'),
    ('Generator RPM, Min (RPM)', 'float64'),
    ('Generator RPM, Standard deviation (RPM)', 'float64'),
    ('Rotor speed, Max (RPM)', 'float64'),
    ('Rotor speed, Min (RPM)', 'float64'),
    ('Rotor speed, Standard deviation (RPM)', 'float64'),
    ('Blade angle (pitch position) (°)', 'float64'),
    ('Blade angle (pitch position), Max (°)', 'float64'),
    ('Blade angle (pitch position), Min (°)', 'float64'),
    ('Blade angle (pitch position), Standard deviation (°)', 'float64'),
    ('Grease left generator bearing', 'float64'),
    ('Grease left shaft bearing', 'float64'),
    ('direccion (grados)', 'float64'),
    ('presion (hPa)', 'float64'),
    ('Xpos', 'int'),
    ('Ypos', 'int'),
    ('Zpos', 'int'),
    ('Turbine', 'U10')  # 'U10' para cadenas de hasta 10 caracteres
]


# Cargar los datos del archivo con punto y coma como delimitadorworkon 
#training_data_not_scaled = np.genfromtxt(datos, delimiter=',', skip_header=1, dtype=dtype, invalid_raise=False)
training_data_not_scaled = pd.read_csv(datos)
print(training_data_not_scaled)



# Acceder a las columnas que necesitas
X = training_data_not_scaled.iloc[:, columnasAUsar]
y = training_data_not_scaled.iloc[:, 4]  # Columna 4 es la variable a predecir



print(X)
print("------------------------")
print(y)

#evaluate model
modelo = evaluate_model(X, y)
#Calor a predecir
valorReferencia=77583
arrayValidas=predict_inverse(valorReferencia,training_data_not_scaled,modelo, posiblesConfiguraciones)
print(arrayValidas)
