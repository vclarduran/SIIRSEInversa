Esta carpeta incluye todos los documentos de código elaborados en la investigación del desarrollo de una "red neuronal inversa" sobre una data set de turbinas aerogeneradoras. 

Incluye:

Set de datos, procesados y programas para su depuración
- Datos unidos calor: set completo de datos
- Preprocesado: programa que elimina los valores nulos del set de datos
- Parcial: parte del data set para poder manipularlo demanera manual y hacer pruebas
- Procesado: set de datos depurado y procesado

Programa de evaluación de la red: 
- redMultiHeatTube: base de la que partimos de proyecto anterior
- pruebasFunciones/paraCrear: pruebas separadas de funcionalidades concretas
- redMultiInverse: programa entero que procesa el set

PASOS QUE EFECTUA EL PROGRAMA/PROCESO LÓGICO: 
OBJETIVO: extraer valores de entradas para un valor de salida concreto
1. Set de los valores de entrada y salida (columnas training, variables a predecir...)
2. Define un cartesiano de todos los valores posibles: crea un nuevo documento en el que habra todos los valores posibles de netrada para todas las variables, para ello, para cada columna de valores de entrada elige
   el mayor y menor valor y escribe todo el rango de entre los dos.
3. Seleciona las filas de datos mas cercanas por encima y debajo del valor que nosotros buscamos (find_nearest y returnValues)
4. Compara los cada una de las entradas de las filas (rowMenor/rowMayor), para crear posibles configuraciones que podrian funcionar (entradasGeneticas):
 - Si los valores son iguales (power[rowUp] == power[rowDown]) el valor de esa entrada será seguro ese (llamamosle asegurada)
 - Si no son iguales, probaremos con todas las entradas posibles creadas en el paso 2 para esa variable de entrada
5. Con las combinaciones elegidas se hace un cartesiano, todas las variables posibles con las aseguradas
6. Se crea y entrena una red neuronal con los datos de partida
7. Se le pasan las combinanciones del cartesiano, si entran dentro de un rango del 0,05% del valor objetivo se elgien como posibles y se muestran por pantalla, sino se descartan
