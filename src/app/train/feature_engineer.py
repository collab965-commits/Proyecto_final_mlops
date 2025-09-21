import pandas as pd
import numpy as np

class FeatureEngineer:
    def __init__(self, df):
        self.df = df

    def create_features(self):
        # 1. Ratio entre varianza y entropía
        self.df["var_entropy_ratio"] = self.df["variance"] / (self.df["entropy"].replace(0, np.nan))

        # 2. Magnitud combinada
        self.df["magnitude"] = np.sqrt(
            self.df["variance"]**2 + self.df["skewness"]**2 + self.df["curtosis"]**2 + self.df["entropy"]**2
        )

        # 3. Bucketización de curtosis
        self.df["bucket_curtosis"] = pd.qcut(self.df["curtosis"], q=3, labels=["low", "medium", "high"])

        # 4. Skewness absoluto
        self.df["abs_skewness"] = self.df["skewness"].abs()

        # 5. Diferencia curtosis - skewness
        self.df["curtosis_minus_skewness"] = self.df["curtosis"] - self.df["skewness"]

        return self.df
