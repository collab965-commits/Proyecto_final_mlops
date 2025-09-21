import warnings
import mlflow
import mlflow.sklearn
import optuna
from optuna.integration.mlflow import MLflowCallback
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings('ignore')


class TrainMlflowOptuna:
    def __init__(
        self, df, numeric_features, categorical_features,
        target_column, model_class=RandomForestClassifier,
        test_size=0.25, model_params=None, mlflow_setup=None,
        n_trials=20, optimization_metric='accuracy',
        param_distributions=None
    ):
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.test_size = test_size
        self.model_class = model_class
        self.model_params = model_params if model_params is not None else {}
        self.n_trials = n_trials
        self.optimization_metric = optimization_metric
        self.param_distributions = param_distributions or {}

        self.setup = mlflow_setup
        self.best_model = None
        self.best_params = None
        self.best_pipeline = None
        self.best_score = None

    def train_test_split(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        return train_test_split(X, y, test_size=self.test_size, random_state=42)

    def create_pipeline_numeric(self):
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])

    def create_pipeline_categorical(self):
        return Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
            ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))
        ])

    def create_preprocessor(self):
        return ColumnTransformer(transformers=[
            ('num', self.create_pipeline_numeric(), self.numeric_features),
            ('cat', self.create_pipeline_categorical(), self.categorical_features)
        ])

    def create_pipeline_train(self, model):
        return Pipeline(steps=[
            ('preprocessor', self.create_preprocessor()),
            ('classifier', model)
        ])

    def create_objective(self, X_train, X_test, y_train, y_test):
        def objective(trial):
            params = {}
            for param_name, param_config in self.param_distributions.items():
                if param_config[0] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config[1], param_config[2], log=param_config[3] if len(param_config) > 3 else False)
                elif param_config[0] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config[1], param_config[2])
                elif param_config[0] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config[1])

            all_params = {**self.model_params, **params}
            model = self.model_class(**all_params)
            pipeline = self.create_pipeline_train(model)
            pipeline.fit(X_train, y_train)

            y_pred = pipeline.predict(X_test)

            if self.optimization_metric == 'accuracy':
                return accuracy_score(y_test, y_pred)
            elif self.optimization_metric == 'f1':
                return f1_score(y_test, y_pred, average='weighted')
            elif self.optimization_metric == 'precision':
                return precision_score(y_test, y_pred, average='weighted', zero_division=0)
            elif self.optimization_metric == 'recall':
                return recall_score(y_test, y_pred, average='weighted', zero_division=0)
            elif self.optimization_metric == 'roc_auc':
                if hasattr(pipeline, 'predict_proba'):
                    y_pred_proba = pipeline.predict_proba(X_test)
                    if y_pred_proba.shape[1] == 2:
                        y_pred_proba = y_pred_proba[:, 1]
                    return roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
                return accuracy_score(y_test, y_pred)

        return objective

    def train_with_optuna(self):
        X_train, X_test, y_train, y_test = self.train_test_split()
        study = optuna.create_study(direction='maximize', study_name=f"optuna_{self.model_class.__name__}")
        objective = self.create_objective(X_train, X_test, y_train, y_test)

        mlflow_callback = MLflowCallback(tracking_uri=mlflow.get_tracking_uri(), metric_name=self.optimization_metric)
        study.optimize(objective, n_trials=self.n_trials, callbacks=[mlflow_callback])

        self.best_params = study.best_params
        self.best_score = study.best_value

        print(f"\nOptimization complete!")
        print(f"Best {self.optimization_metric}: {self.best_score:.4f}")
        print(f"Best parameters: {self.best_params}")

        best_run_id = self._train_best_model(X_train, X_test, y_train, y_test)
        return self.best_pipeline, best_run_id, study

    def _train_best_model(self, X_train, X_test, y_train, y_test):
        with mlflow.start_run(run_name=f"best_model_{self.model_class.__name__}") as run:
            all_params = {**self.model_params, **self.best_params}
            self.best_model = self.model_class(**all_params)
            self.best_pipeline = self.create_pipeline_train(self.best_model)
            self.best_pipeline.fit(X_train, y_train)

            y_train_pred = self.best_pipeline.predict(X_train)
            y_test_pred = self.best_pipeline.predict(X_test)

            mlflow.log_metric("train_accuracy", accuracy_score(y_train, y_train_pred))
            mlflow.log_metric("test_accuracy", accuracy_score(y_test, y_test_pred))

            mlflow.sklearn.log_model(self.best_pipeline, "model")

            run_id = run.info.run_id
            print(f"\nBest Model MLflow Run ID: {run_id}")
            print(f"Tracking URI: {mlflow.get_tracking_uri()}")

            return run_id

    def train(self):
        return self.train_with_optuna()