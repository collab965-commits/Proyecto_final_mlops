# ======================================================
# 🚀 Dockerfile - Proyecto MLOps: Autenticación de Billetes
# ======================================================

# Imagen base
FROM python:3.11-slim

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copiar archivos de configuración
COPY pyproject.toml uv.lock ./

# ----------------------------------------------------------
# Instalar dependencias con más tolerancia a red
# ----------------------------------------------------------
ENV UV_HTTP_TIMEOUT=120
ENV UV_HTTP_MAX_RETRIES=5

# Instalar dependencias (si uv falla, usar pip)
RUN pip install --no-cache-dir -r pyproject.toml || true

# Instalar MLflow y componentes
RUN pip install --no-cache-dir \
    mlflow==3.4.0 \
    scikit-learn==1.4.2 \
    optuna==3.6.1 \
    pandas \
    numpy \
    matplotlib \
    gunicorn

# Copiar el código fuente del proyecto
COPY src/ ./src/
COPY notebooks/ ./notebooks/

# Crear directorios para modelos y artefactos
RUN mkdir -p /app/data /app/models /app/mlruns /app/mlartifacts

# Variables de entorno
ENV PYTHONPATH=/app
EXPOSE 5000 8888

# Comando por defecto
CMD ["python", "-m", "src.app.train.task_train"]
