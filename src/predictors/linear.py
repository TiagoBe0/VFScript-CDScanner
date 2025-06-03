# modifiers/predictors/linear.py

import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils.io_utils import load_json_data

class VacancyPredictor:
    """
    Regresión lineal simple (usa solo 'surface_area' para predecir vacancys).
    """
    def __init__(self, json_path: str = "outputs.vfinder/training_data.json"):
        self.json_path = json_path
        self.columns = ["surface_area"]
        self.df = load_json_data(self.json_path)
        self.model = self._train_model()

    def _train_model(self):
        X = self.df[self.columns]
        y = self.df["vacancys"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        # (Opcional) imprimir o registrar mse
        return model

    @staticmethod
    def _round_positive(x):
        return math.ceil(x) if x > 0 else math.ceil(-x)

    def predict_vacancies(self, **kwargs):
        nuevos_datos = pd.DataFrame({col: [kwargs[col]] for col in self.columns})
        prediction = self.model.predict(nuevos_datos)[0]
        return self._round_positive(prediction)
