import pandas as pd
import os

columnasAUsar = [0,1,2,3,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]

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

    print("End of valoresPosibles")
    return variablesPosibles



array = valoresPosibles("parcial.csv")



	arrays = [np.asarray(x) for x in arrays]
	dtype = arrays[0].dtype
	print(dtype)
	
	n = np.prod([x.size for x in arrays])
	print(n)
	if(n<0): n = n*(-1)
	if out is None:
		out = np.zeros([n, len(arrays)], dtype=np.float64)
	
	m = n // arrays[0].size
	out[:, 0] = np.repeat(arrays[0], m)
	if arrays[1:]:
		cartesian(arrays[1:], out=out[:m, 1:])
		for j in range(1, arrays[0].size):
			out[j * m:(j + 1) * m, 1:] = out[:m, 1:]
	
	print(out)
	return out