# modifiers/predictors/random_forest.py

import math
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from utils.io_utils import load_json_data

class VacancyPredictorRF:
    """
    Random Forest: usa las columnas definidas en PREDICTOR_COLUMNS para predecir vacancys.
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
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        # (Opcional) registrar mse
        return model

    @staticmethod
    def _round_up(x):
        return math.ceil(x)

    def predict_vacancies(self, **kwargs):
        data = {col: [kwargs[col]] for col in self.columns}
        nuevos_datos = pd.DataFrame(data)
        prediction = self.model.predict(nuevos_datos)[0]
        return self._round_up(prediction)
