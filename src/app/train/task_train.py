from sklearn.linear_model import LogisticRegression
from src.app.train.etl import UserGenerator   # carga de dataset
from src.app.train.feature_engineer import FeatureEngineer  # ingeniería de features
from src.app.train.train import Train  # entrenamiento del modelo
# Si tuvieras MLflow configurado, usarías TrainWithMlflow, pero lo dejamos fuera por ahora


def task_train():
    # 1) Cargar datos
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
    user_generator = UserGenerator(url)   # en tu versión UserGenerator recibe url
    df = user_generator.create_dataset()

    # 2) Ingeniería de características
    feature_engineer = FeatureEngineer(df)
    df_engineered = feature_engineer.create_features()

    # 3) Configuración de features y target
    numeric_features = ["variance", "skewness", "curtosis", "entropy",
                        "var_entropy_ratio", "magnitude", "abs_skewness", "curtosis_minus_skewness"]
    categorical_features = ["bucket_curtosis"]  # categórica creada en FeatureEngineer
    target_column = "class"
    test_size = 0.25
    model = LogisticRegression(random_state=42, max_iter=1000)

    # 4) Entrenamiento
    train = Train(df_engineered, numeric_features, categorical_features, target_column, model, test_size)
    pipeline, X_test, y_test = train.train()

    return pipeline, X_test, y_test
