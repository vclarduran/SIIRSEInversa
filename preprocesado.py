import csv
import os
import itertools

def procesar_y_escribir_linea(ruta_entrada, ruta_salida, num_lineas=100000):
    with open(ruta_entrada, newline='') as entrada, open(ruta_salida, 'w', newline='') as salida:
        lector_csv = csv.reader(itertools.islice(entrada, num_lineas))
        escritor_csv = csv.writer(salida)
        
        for fila in lector_csv:
            # Quitar la primera y última instancia de la lista
            fila.pop(0)
            fila.pop(-1)
            
            # Verificar si hay algún valor nulo en la fila
            if None not in fila:
                # Escribir la fila procesada en el archivo de salida
                escritor_csv.writerow(fila)

nombre_archivo_entrada = "/home/laboratorio-ss-03/Escritorio/completo.csv"
nombre_archivo_salida = "/home/laboratorio-ss-03/Escritorio/procesado.csv"

procesar_y_escribir_linea(nombre_archivo_entrada, nombre_archivo_salida, num_lineas=100000)
