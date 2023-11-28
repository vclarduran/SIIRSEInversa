# use mlp for prediction on multi-output regression
from numpy import asarray
from sklearn.datasets import make_regression
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import math
import matplotlib.pyplot as plt
from keras.layers.core import Dense
import tensorflow as tf
from numpy import genfromtxt
from keras.layers import Dropout

from random import random
from numpy import array
from numpy import cumsum
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import RepeatedKFold

from concurrent.futures import ProcessPoolExecutor

import csv

import pandas as pd
import os
import sys
from itertools import product

datos = "parcial.csv" #CAMBIAR EL CSV
columnasAUsar = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
variablesAPredecir = 1
columnaPredecida = 4
#COLUMNA 4 = Power



def cartesian(arrays):
    output_csv = "output.csv"
    print("Entering cartesian_to_csv function")

    # Get the current working directory
    current_directory = os.getcwd()
    print(f"Current working directory: {current_directory}")

    # Combine the current directory with the output CSV file name
    output_path = os.path.join(current_directory, output_csv)
    print(f"Output path for CSV file: {output_path}")

    # Write the Cartesian product directly to the CSV file
    with open(output_path, 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)

        # Write header based on the number of arrays
        header_row = [f'Array_{i+1}' for i in range(len(arrays))]
        print(f"Writing header row: {header_row}")
        csv_writer.writerow(header_row)

        # Generate and write each row of the Cartesian product
        for cartesian_row in product(*arrays):
            print(f"Writing row: {cartesian_row}")
            csv_writer.writerow(cartesian_row)

    print(f"CSV file '{output_path}' created")
    return output_path



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

	rowMenor=data.iloc[indexMenor,0:25]
	rowMayor=data.iloc[indexMayor,0:25]
	return rowMenor, rowMayor

def generarEntradasGeneticas(rowMenor, rowMayor, variablesPosibles):
	arrayValidacion=[]

	for columna in columnasAUsar:
		arrayColumna =[]
		print(columna)
		if(rowMayor.iloc[columna] == rowMenor.iloc[columna]):
			arrayColumna.append(rowMayor.iloc[columna])
			arrayValidacion.append(arrayColumna)
		else:
			if(columna>4):
				arrayValidacion.append(variablesPosibles[columna-1])

			else:
				arrayValidacion.append(variablesPosibles[columna])

	filename = cartesian(arrayValidacion)
	return filename

def get_model(n_inputs, n_outputs): #Crea un modelo de dos capas 
	model = Sequential()
	model.add(Dense(20, input_dim=n_inputs, kernel_initializer='he_uniform', activation='relu'))
	model.add(Dense(13626, activation='relu'))

	model.add(Dense(n_outputs, activation="linear"))
	model.compile(loss='mse', optimizer='adam',metrics=['mse','mae','accuracy'])
	return model

def predict_inverse(valorReferencia, training_data_not_scaled, model, posiblesConfiguraciones):
    # valorReferencia = 77583
    indexMenor, indexMayor = find_nearest(training_data_not_scaled, valorReferencia, columnaPredecida)

    rowDown, rowUp = returnValues(training_data_not_scaled, indexMenor, indexMayor)
    file = generarEntradasGeneticas(rowDown, rowUp, posiblesConfiguraciones)

    print("generadas")

    scaler_x = MinMaxScaler() 
    scaler_y = MinMaxScaler()
    print("despues de MinMaxScaler")

    arrayValidas = []

    print(f"Trying to open file: {file}")
    with open(file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for linea in csv_reader:
            newX = asarray([linea])
            print(linea)
            geneticas_scaled = scaler_x.transform(newX)
            yhat = model.predict(geneticas_scaled)
            yInverse = scaler_y.inverse_transform(yhat)

            if (yInverse[0][0] > (valorReferencia * 0.95) and yInverse[0][0] < (valorReferencia * 1.05) and
                    yInverse[0][1] > 0 and yInverse[0][1] < 1 and yInverse[0][2] > 0 and yInverse[0][2] < 1):
                geneticaSinEscalar = newX
                listaGenetica = geneticaSinEscalar[0].tolist()
                listaGenetica.append(yInverse[0][0])
                listaGenetica.append(yInverse[0][1])
                listaGenetica.append(yInverse[0][2])
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
    print("Start of valoresPosibles")

    # Load CSV
    df = pd.read_csv(os.path.join(datos), delimiter=",")

    # Find min and max
    minimos = df.min()
    maximos = df.max()

    # Initialize list for possible values
    variablesPosibles = []

    for columna in columnasAUsar:
        min_value = round((minimos.iloc[columna] * 0.95), 2)
        max_value = round((maximos.iloc[columna] * 1.05), 2)
        valor = min_value
        valoresDeVariable = [valor]

        print(f"Columna: {columna}, Min: {min_value}, Max: {max_value}")

        while valor < max_value:
            valoresDeVariable.append(round(valor, 2))
            valor += 0.01

        print(f"Number of iterations for Columna {columna}: {len(valoresDeVariable)}")
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
    ('Turbine', 'U10') 
]


# Cargar los datos del archivo con punto y coma como delimitadorworkon 
#training_data_not_scaled = np.genfromtxt(datos, delimiter=',', skip_header=1, dtype=dtype, invalid_raise=False)
print("1")
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
