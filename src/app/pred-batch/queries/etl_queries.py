#!/usr/bin/env python3
"""
Script para extraer datos desde una base de datos PostgreSQL para el flujo de predicciones.
Adaptado al proyecto de Julián, siguiendo la estructura estandarizada del pipeline.
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

# ======================================================
# 1️⃣ Cargar variables de entorno (opcional, si usas .env)
# ======================================================
load_dotenv()

# ======================================================
# 2️⃣ Configuración de conexión a la base de datos
# ======================================================
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "postgres")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "postgres")

# Crear el motor SQLAlchemy
CONN_STR = f"postgresql://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(CONN_STR, echo=False)


# ======================================================
# 3️⃣ Definir las consultas SQL
# ======================================================
def get_pred_query():
    """
    Devuelve la consulta SQL utilizada para extraer los datos de predicción.
    Puedes modificar esta query según tus tablas y campos.
    """
    query = """
        SELECT 
            id_cliente,
            edad,
            ingresos_mensuales,
            antiguedad_cliente,
            numero_transacciones,
            monto_promedio,
            tipo_producto,
            region,
            canal_preferido,
            tiene_credito,
            mora_actual,
            score_crediticio
        FROM 
            schema_predicciones.clientes_prediccion
        WHERE 
            activo = TRUE;
    """
    return query


# ======================================================
# 4️⃣ Función principal del ETL
# ======================================================
def etl_queries():
    """
    Ejecuta el proceso ETL:
      - Ejecuta la query de predicciones
      - Carga los datos en un DataFrame de pandas
    """
    print("\n🚀 Iniciando proceso ETL desde base de datos PostgreSQL...\n")

    query = get_pred_query()

    try:
        df = pd.read_sql(query, engine)
        print(f"✅ Datos extraídos correctamente: {df.shape[0]} filas × {df.shape[1]} columnas.")
        print(f"📋 Columnas: {list(df.columns)}")
        return df

    except Exception as e:
        print(f"❌ Error al ejecutar la query: {e}")
        return pd.DataFrame()  # Devuelve vacío para evitar ruptura del pipeline


# ======================================================
# 5️⃣ Ejecución directa (modo notebook o script)
# ======================================================
if __name__ == "__main__":
    df = etl_queries()
    if not df.empty:
        print("\n🧾 Vista previa de los datos extraídos:")
        print(df.head())
