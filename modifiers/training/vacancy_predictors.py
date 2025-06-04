# modifiers/training/vacancy_predictors.py

import os
import json
import math
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import xgboost as xgb
from training.utils import load_json_data


class VacancyPredictorRF:
    def __init__(
        self,
        json_path: str = "outputs.vfinder/training_data.json",
        predictor_columns: list = None
    ):
        """
        Predictor de vacancias con RandomForest. Usa las columnas que se definen
        en predictor_columns (se obtienen de input_params.json).
        """
        self.json_path = json_path

        # Si predictor_columns no se pasa explícito, lo extraemos de input_params.json
        if predictor_columns is None:
            # si quisieras leer predictor_columns de input_params.json:
            raise ValueError("Debes pasar predictor_columns o definirlas explícitamente.")
        self.columns = predictor_columns

        # Cargamos DataFrame
        self.df = load_json_data(self.json_path)

        # Entrenamos el modelo
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
        # Opcionalmente podrías imprimir o guardar 'mse'
        return model


    @staticmethod
    def _round_up(x):
        return math.ceil(x)


    def predict_vacancies(self, **kwargs):
        data = {col: [kwargs[col]] for col in self.columns}
        nuevos_datos = pd.DataFrame(data)
        prediction = self.model.predict(nuevos_datos)[0]
        print(f"Predicción inicial: {prediction}")
        return self._round_up(prediction)




class XGBoostVacancyPredictor:
    def __init__(
        self,
        training_data_path: str = "outputs.vfinder/training_data.json",
        model_path: str = "outputs.json/xgboost_model.json",
        predictor_columns: list = None,
        n_splits: int = 5,
        random_state: int = 42
    ):
        """
        Predictor de vacancias con XGBoost. Hace cross‐validation y guarda el modelo
        en 'model_path'.
        """
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.n_splits = n_splits
        self.random_state = random_state

        if predictor_columns is None:
            raise ValueError("Debes pasar predictor_columns o definirlas explícitamente.")
        self.columns = predictor_columns

        self.scaler = StandardScaler()
        self.model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=self.random_state,
            n_estimators=100,
            learning_rate=0.1,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8
        )
        self._load_data_and_train()


    def _load_data_and_train(self):
        with open(self.training_data_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        feature_list = []
        for col in self.columns:
            if col in data:
                feature_list.append(data[col])
            else:
                raise ValueError(f"No existe la columna '{col}' en los datos de entrenamiento.")

        X = np.column_stack(feature_list)
        y = np.array(data["vacancys"])

        X = self.scaler.fit_transform(X)
        # Ajustamos número de folds según cantidad de muestras
        n_samples = X.shape[0]
        n_splits = self.n_splits if n_samples >= self.n_splits else max(n_samples, 2)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)

        scores = cross_val_score(self.model, X, y, scoring='neg_mean_squared_error', cv=kfold)
        mse_scores = -scores  # Convertir a positivos
        # Opcionalmente podrías imprimir o registrar mse_scores

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        # Guardamos el modelo en disco
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)


    def predict(self, sample_input: np.ndarray) -> np.ndarray:
        sample_input = np.array(sample_input)
        sample_input = self.scaler.transform(sample_input)
        prediction = self.model.predict(sample_input)
        print(f"Predicción inicial: {prediction}")
        return prediction




class VacancyPredictor:
    def __init__(self, json_path: str = "outputs.vfinder/training_data.json"):
        """
        Predictor de vacancias con regresión lineal simple, usando solo 'surface_area'.
        """
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
        # Opcionalmente imprimir o registrar mse
        return model


    @staticmethod
    def _round_positive(x):
        return math.ceil(x) if x > 0 else math.ceil(-x)


    def predict_vacancies(self, **kwargs):
        nuevos_datos = pd.DataFrame({col: [kwargs[col]] for col in self.columns})
        prediction = self.model.predict(nuevos_datos)[0]
        print(f"Predicción inicial: {prediction}")
        return self._round_positive(prediction)




from sklearn.neural_network import MLPRegressor

class VacancyPredictorMLP:
    def __init__(
        self,
        json_path: str = "outputs.vfinder/training_data.json",
        predictor_columns: list = None
    ):
        """
        Predictor de vacancias con red neuronal (MLP). Crea un pipeline que escala
        y entrena un MLPRegressor.
        """
        self.json_path = json_path

        if predictor_columns is None:
            raise ValueError("Debes pasar predictor_columns o definirlas explícitamente.")
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
        print("MSE del modelo MLP:", mse)

        return pipeline


    @staticmethod
    def _round_up(x):
        return math.ceil(x) if x > 0 else math.ceil(-x)


    def predict_vacancies(self, **kwargs):
        data = pd.DataFrame({col: [kwargs[col]] for col in self.columns})
        prediction = self.model.predict(data)[0]
        print(f"Predicción inicial: {prediction}")  
        return self._round_up(prediction)
