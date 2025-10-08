#!/usr/bin/env python3
"""
Script de predicción del modelo de autenticación de billetes.
Replica la lógica de prediction.ipynb, estructurada por etapas:
1️⃣ Configuración de rutas
2️⃣ Carga del modelo
3️⃣ Extracción de datos (ETL)
4️⃣ Ingeniería de features
5️⃣ Generación de predicciones
6️⃣ Guardado de resultados
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd
import urllib.request

# ============================================
# 1️⃣ CONFIGURACIÓN DE RUTAS
# ============================================

# Ruta base (sube 3 niveles desde queries/)
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), '..', '..', '..'))
SRC_DIR = os.path.join(BASE_DIR, 'src')

# Agregar 'src' al sys.path si no está
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

print(f"✅ sys.path incluye: {SRC_DIR}")

# ============================================
# 2️⃣ IMPORTAR MÓDULOS INTERNOS
# ============================================

try:
    from etl_pred import UserGenerator
    from feature_engineer_pred import FeatureEngineer
    print("✅ Módulos internos importados correctamente.")
except ModuleNotFoundError as e:
    print("❌ Error al importar módulos:", e)
    sys.exit(1)

# ============================================
# 3️⃣ FUNCIÓN PARA CARGAR EL MODELO
# ============================================
def get_model():
    # ✅ Ruta correcta: el modelo está en src/train/models/model_rf.pkl
    model_path = os.path.abspath(
        os.path.join(SRC_DIR, "train", "models", "model_rf.pkl")
    )

    print(f"📁 Buscando modelo en: {model_path}")

    # Validar si existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ No se encontró el modelo en {model_path}")

    # Cargar modelo
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    print(f"✅ Modelo cargado correctamente desde: {model_path}")
    return model


# ============================================
# 4️⃣ EXTRACCIÓN DE DATOS (ETL)
# ============================================

def get_etl_data():
    data_dir = os.path.join(SRC_DIR, "train", "data")
    os.makedirs(data_dir, exist_ok=True)

    data_path = os.path.join(data_dir, "data_banknote_authentication.txt")

    # Descargar dataset si no existe
    if not os.path.exists(data_path):
        print("🌐 Descargando dataset desde UCI Repository...")
        data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
        urllib.request.urlretrieve(data_url, data_path)
        print(f"✅ Dataset descargado en: {data_path}")

    user_generator = UserGenerator(url=data_path)
    df = user_generator.create_dataset()

    print(f"✅ Dataset cargado correctamente: {df.shape[0]} filas, {df.shape[1]} columnas.")
    return df

# ============================================
# 5️⃣ INGENIERÍA DE FEATURES
# ============================================

def get_data():
    df = get_etl_data()
    feature_engineer = FeatureEngineer(df)
    df = feature_engineer.create_features()

    # Asegurar columnas requeridas por el modelo
    if 'abs_skewness' not in df.columns:
        df['abs_skewness'] = np.abs(df['skewness'])
    
    if 'var_entropy_ratio' not in df.columns:
        df['var_entropy_ratio'] = np.where(
            df['entropy'] != 0,
            df['variance'] / df['entropy'],
            0
        )

    if 'bucket_curtosis' not in df.columns:
        df['bucket_curtosis'] = pd.cut(
            df['curtosis'],
            bins=3,
            labels=['low', 'medium', 'high']
        )

    print("✅ Features generadas correctamente. Total columnas:", len(df.columns))
    print("📋 Columnas finales:", list(df.columns))
    return df

# ============================================
# 6️⃣ PREDICCIÓN
# ============================================

def predict(model, df):
    prediction = model.predict(df)
    print("✅ Predicciones generadas correctamente.")
    return prediction

# ============================================
# 7️⃣ GUARDAR PREDICCIONES
# ============================================

def save_prediction(df, prediction):
    output_dir = os.path.join(SRC_DIR, "app", "pred-batch", "queries", "predictions")
    os.makedirs(output_dir, exist_ok=True)

    df["prediction"] = prediction
    df["prediction"] = df["prediction"].astype(int)
    df["prediction"] = df["prediction"].map({0: "No", 1: "Sí"})

    output_path = os.path.join(output_dir, "predictions.csv")
    df.to_csv(output_path, index=False, encoding="utf-8")

    print(f"✅ Predicciones guardadas en: {output_path}")
    return df

# ============================================
# 8️⃣ MAIN - EJECUCIÓN COMPLETA
# ============================================

def main():
    print("\n🚀 Iniciando proceso de predicción...\n")

    model = get_model()
    df = get_data()

    if "class" in df.columns:
        df = df.drop(columns=["class"])

    prediction = predict(model, df)
    save_prediction(df, prediction)

    print("\n🎯 Proceso completado exitosamente.\n")

# ============================================
# 9️⃣ PUNTO DE ENTRADA
# ============================================

if __name__ == "__main__":
    main()
