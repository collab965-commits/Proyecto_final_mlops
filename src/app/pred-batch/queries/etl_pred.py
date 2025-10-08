#!/usr/bin/env python3
"""
Script para cargar y limpiar el dataset de autenticación de billetes.
Adaptado al flujo del proyecto de Julián, siguiendo el estilo del profesor.
"""

import os
import pandas as pd
import numpy as np


class UserGenerator:
    def __init__(self, url, seed=42):
        """
        Inicializa el generador de datos reales de billetes.

        Args:
            url (str): Ruta local del dataset (archivo .txt)
            seed (int): Semilla para reproducibilidad
        """
        self.url = url
        self.seed = seed
        self.cols = ["variance", "skewness", "curtosis", "entropy", "class"]

    # =======================================================
    # 1️⃣ Cargar dataset real
    # =======================================================
    def generate_synthetic_users(self):
        """
        Carga el dataset real de billetes desde la ruta indicada.
        """
        print("\n📂 Cargando dataset de billetes...")
        try:
            df = pd.read_csv(self.url, header=None, names=self.cols)
            print(f"✅ Dataset cargado correctamente: {len(df)} filas, {len(df.columns)} columnas.")
        except FileNotFoundError:
            raise FileNotFoundError(f"❌ No se encontró el archivo en la ruta: {self.url}")
        return df

    # =======================================================
    # 2️⃣ Revisar y limpiar datos faltantes
    # =======================================================
    def add_missing_data(self, df):
        """
        Revisa si hay nulos y limpia los registros si existen.
        """
        print("\n🧹 Revisando datos nulos...")
        null_counts = df.isna().sum()
        print(null_counts)
        if null_counts.sum() > 0:
            df = df.dropna().copy()
            print("✅ Se eliminaron filas con valores nulos.")
        else:
            print("✅ No se encontraron valores nulos.")
        return df

    # =======================================================
    # 3️⃣ Crear dataset final
    # =======================================================
    def create_dataset(self):
        """
        Carga, limpia y devuelve el dataset de billetes autenticados.
        """
        print("\n🚀 Generando dataset de autenticación de billetes...\n")

        # Validar ruta
        if not os.path.exists(self.url):
            raise FileNotFoundError(f"❌ No se encontró el dataset en: {self.url}")

        # Cargar dataset real
        df = self.generate_synthetic_users()

        # Limpiar datos
        df = self.add_missing_data(df)

        # Revisar distribución de la clase
        print("\n📊 Distribución de la clase (0=auténtico, 1=falso):")
        print(df["class"].value_counts())
        print("\n📈 Proporción:")
        print(df["class"].value_counts(normalize=True).round(3))

        print("\n✅ Dataset final listo para procesamiento.")
        return df
