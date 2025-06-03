# modifiers/predictors/mlp.py

import math
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils.io_utils import load_json_data

class VacancyPredictorMLP:
    """
    Red neuronal MLP con escalado previo. Usa las columnas de PREDICTOR_COLUMNS.
    """
    def __init__(
        self,
        json_path: str = "outputs.vfinder/training_data.json",
        predictor_columns: list = None
    ):
        self.json_path = json_path
        if predictor_columns is None:
            raise ValueError("Debes pasar predictor_columns explícitamente.")
        self.columns = predictor_columns
        self.df = load_json_data(self.json_path)
        self.model = self._train_model()

    def _train_model(self):
        X = self.df[self.columns]
        y = self.df["vacancys"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                learning_rate_init=0.01,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42
            ))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print(f"[MLP] MSE del modelo: {mse}")
        return pipeline

    @staticmethod
    def _round_up(x):
        return math.ceil(x)

    def predict_vacancies(self, **kwargs):
        data = pd.DataFrame({col: [kwargs[col]] for col in self.columns})
        prediction = self.model.predict(data)[0]
        return self._round_up(prediction)
