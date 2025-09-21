from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
import mlflow
import mlflow.sklearn
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score
)
import warnings
warnings.filterwarnings('ignore')


class TrainMlflow:
    def __init__(
        self, df, numeric_features, target_column, model,
        test_size=0.2, model_params=None, mlflow_setup=None
    ):
        """
        Clase para entrenamiento con MLflow adaptada al dataset de billetes.

        Args:
            df: Input dataframe
            numeric_features: Lista de columnas numéricas
            target_column: Nombre de la columna target
            model: Modelo sklearn (ej. RandomForestClassifier)
            model_params: Diccionario de hiperparámetros
            test_size: Proporción de test
            mlflow_setup: configuración de mlflow
        """
        self.df = df
        self.numeric_features = numeric_features
        self.target_column = target_column
        self.test_size = test_size
        self.model = model
        self.model_params = model_params if model_params is not None else {}
        self.setup = mlflow_setup

    def train_test_split(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def create_pipeline_numeric(self):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        return numeric_transformer

    def create_preprocessor(self):
        # Solo numéricas (no hay categóricas en el dataset de billetes)
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', self.create_pipeline_numeric(), self.numeric_features)
            ]
        )
        return preprocessor

    def create_pipeline_train(self):
        pipeline = Pipeline(steps=[
            ('preprocessor', self.create_preprocessor()),
            ('classifier', self.model)
        ])
        return pipeline

    def train(self):
        """
        Entrena el modelo con tracking en MLflow.
        """
        with mlflow.start_run() as run:
            # Log básicos
            mlflow.log_param("test_size", self.test_size)
            mlflow.log_param("model_type", type(self.model).__name__)
            mlflow.log_param("n_numeric_features", len(self.numeric_features))
            mlflow.log_param("target_column", self.target_column)

            # Log hiperparámetros
            for param_name, param_value in self.model_params.items():
                mlflow.log_param(f"model_{param_name}", param_value)

            # Log nombres de features
            mlflow.log_param("numeric_features", ", ".join(self.numeric_features))

            # Split
            X_train, X_test, y_train, y_test = self.train_test_split()

            # Log tamaños
            mlflow.log_param("n_train_samples", len(X_train))
            mlflow.log_param("n_test_samples", len(X_test))

            # Pipeline
            pipeline = self.create_pipeline_train()
            pipeline.fit(X_train, y_train)

            # Predicciones
            y_train_pred = pipeline.predict(X_train)
            y_test_pred = pipeline.predict(X_test)

            # Métricas
            train_metrics = self._calculate_metrics(y_train, y_train_pred, prefix="train")
            test_metrics = self._calculate_metrics(y_test, y_test_pred, prefix="test")

            all_metrics = {**train_metrics, **test_metrics}
            for metric_name, metric_value in all_metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            run_id = run.info.run_id

            print(f"MLflow Run ID: {run_id}")
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")
            print(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
            print(f"Test Accuracy: {test_metrics['test_accuracy']:.4f}")

            return pipeline, run_id

    def _calculate_metrics(self, y_true, y_pred, prefix=""):
        metrics = {}
        accuracy = accuracy_score(y_true, y_pred)
        metrics[f"{prefix}_accuracy"] = accuracy
        try:
            precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics[f"{prefix}_precision"] = precision
            metrics[f"{prefix}_recall"] = recall
            metrics[f"{prefix}_f1"] = f1
        except Exception:
            pass
        return metrics
