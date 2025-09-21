from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class Train:
    def __init__(self, df, numeric_features, categorical_features, target_column, model, test_size=0.2):
        self.df = df
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_column = target_column
        self.model = model
        self.test_size = test_size

    def train_test_split(self):
        X = self.df.drop(columns=[self.target_column])
        y = self.df[self.target_column]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42
        )
        return X_train, X_test, y_train, y_test

    def create_pipeline_numeric(self):
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

    def create_pipeline_categorical(self):
        return Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
            ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ])

    def create_preprocessor(self):
        return ColumnTransformer(
            transformers=[
                ("num", self.create_pipeline_numeric(), self.numeric_features),
                ("cat", self.create_pipeline_categorical(), self.categorical_features),
            ]
        )

    def create_pipeline_train(self):
        return Pipeline(steps=[
            ("preprocessor", self.create_preprocessor()),
            ("classifier", self.model)
        ])

    def train(self):
        X_train, X_test, y_train, y_test = self.train_test_split()
        pipeline = self.create_pipeline_train()
        pipeline.fit(X_train, y_train)
        return pipeline, X_test, y_test
