import csv

nombre_archivo_entrada = "archivo.csv"
nombre_archivo_salida = "archivo_procesado.csv"

with open(nombre_archivo_entrada, newline='') as entrada, open(nombre_archivo_salida, 'w', newline='') as salida:
        lector_csv = csv.reader(entrada)
        escritor_csv = csv.writer(salida)
        
        for linea in lector_csv:
            # Convertir la línea en una lista
            lista = linea.split(',')
            
            # Eliminar la primera instancia
            lista.pop(0)
            
            # Escribir la línea procesada en el archivo de salida
            escritor_csv.writerow(lista)
