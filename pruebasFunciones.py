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



def find_nearest(array,valor):
	mayor=sys.maxsize
	menor=0
	idxMayor=0
	idxMenor=0
	for i in range(0, len(array)):
		if (array.iloc[i][12]>menor and array.iloc[i][12]<=(valor*1)):
			menor=array.iloc[i][12]
			idxMenor=i
		if (array.iloc[i][12]<mayor and array.iloc[i][12]>=(valor*1)):
			mayor=array.iloc[i][12]
			idxMayor=i
	return idxMenor, idxMayor




 arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype
    print(dtype)

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=np.float64)

    m = n // arrays[0].size
    out[:, 0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[:m, 1:])
        for j in range(1, arrays[0].size):
            out[j * m:(j + 1) * m, 1:] = out[:m, 1:]