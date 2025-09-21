import pandas as pd

class UserGenerator:
    def __init__(self, url, seed=42):
        self.url = url
        self.seed = seed
        self.cols = ["variance", "skewness", "curtosis", "entropy", "class"]

    def generate_synthetic_users(self):
        """Carga el dataset real de billetes desde UCI."""
        df = pd.read_csv(self.url, header=None, names=self.cols)
        return df

    def add_missing_data(self, df):
        """Revisa si hay nulos y limpia si los hubiera."""
        print("\n# Nulos por columna")
        print(df.isna().sum())
        return df.dropna().copy()

    def create_dataset(self):
        """Función principal: carga, limpia y devuelve el dataset."""
        print("📥 Cargando dataset de billetes...")

        df = self.generate_synthetic_users()
        df = self.add_missing_data(df)

        print("\n# Distribución de la clase (0=auténtico, 1=falso)")
        print(df["class"].value_counts())
        print("\nProporción:")
        print(df["class"].value_counts(normalize=True).round(3))

        return df
