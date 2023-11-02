import pandas as pd
import os


#lectura csv
df = pd.read_csv(os.path.join("datosUnidosCalor.csv"), delimiter=",", header=None)

#buscar min y max
minimos = df.min()
maximos = df.max()

#para cada columna

columnasAUsar=[1,2,3,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
todosValores =[]

for columna in columnasAUsar:
    min = round((minimos[columna]*0.95),12)
    max = round((maximos[columna]*1.05),12) 
    valor = min
    valoresPosibles = []
    valoresPosibles.append(valor) 
    while(valor < max):
        valoresPosibles.append(round(valor,12))
        valor+=0.1
    todosValores.append(valoresPosibles)    
    



