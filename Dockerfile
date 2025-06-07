# Imagen base
FROM python:3.11-slim

# Instalamos compiladores y BLAS/LAPACK  
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      build-essential gfortran libopenblas-dev liblapack-dev && \
    rm -rf /var/lib/apt/lists/*

# Subimos el timeout y actualizamos pip  
ENV PIP_DEFAULT_TIMEOUT=120  
RUN pip install --upgrade pip setuptools wheel

# Directorio de trabajo
WORKDIR /app

# Copiar ficheros de código y dependencias
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app .

# Exponer el puerto de Streamlit
EXPOSE 8501

# Comando de arranque
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
