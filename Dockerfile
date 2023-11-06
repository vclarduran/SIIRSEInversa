FROM amdih/tensorflow:rocm5.0-tf2.7-dev

# Actualizar el sistema operativo y configurar el directorio de trabajo
RUN apt-get update && apt-get install -y python3-pip
RUN pip3 install matplotlib
RUN apt-get update -y
RUN apt-get install -y libx11-dev
RUN apt-get install -y python3-tk
RUN pip install requests

WORKDIR /app

# Copiar los archivos de requerimientos y los scripts de Python
COPY requirements.txt .
COPY *.py ./
COPY *.csv ./

# Instalar las dependencias especificadas en el archivo de requerimientos
RUN pip3 install --no-cache-dir -r requirements.txt

# Ejecutar el script principal de Python

CMD ["python3", "redMultiInverse.py"]

# docker build -t (nombre_de_la_imagen) .
# docker run -it (nombre_de_la_imagen)
