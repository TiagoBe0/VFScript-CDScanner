# modifiers/training/vacancy_predictor_curve.py

import os
import json
import numpy as np
import pandas as pd
from scipy.optimize import brentq

class VacancyPredictorCurve:
    def __init__(self, training_json_path: str, csv_path: str, degree: int = 3):
        """
        - training_json_path: ruta a un JSON con datos de entrenamiento (vacancys y surface_area).
        - csv_path: ruta a un CSV de entrada con columna "area".
        - degree: grado del polinomio a ajustar.
        """
        self.training_json_path = training_json_path
        self.csv_path = csv_path
        self.degree = degree

        self.training_data = None
        self.vacancias_train = None
        self.surface_area_train = None
        self.poly = None
        self.min_area_train = None
        self.max_area_train = None


    def load_training_data(self, as_dataframe: bool = False):
        """
        Carga datos de entrenamiento desde el JSON. Si as_dataframe=True, devuelve DataFrame.
        """
        with open(self.training_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if as_dataframe:
            data = pd.DataFrame(data)

        self.training_data = data
        return self.training_data


    def prepare_training_data(self):
        """
        Extrae las series de vacancias y surface_area (saltándose los primeros 2 datos)
        para luego definir el rango de ajuste.
        """
        if self.training_data is None:
            raise ValueError("Debes llamar a load_training_data() primero.")

        # Convertir a DataFrame si no lo es
        df = pd.DataFrame(self.training_data) if not isinstance(self.training_data, pd.DataFrame) else self.training_data

        # Tomamos desde el índice 2 en adelante, tal como en el script original
        self.vacancias_train     = df["vacancys"].iloc[2:]
        self.surface_area_train  = df["surface_area"].iloc[2:]
        self.min_area_train      = self.surface_area_train.min()
        self.max_area_train      = self.surface_area_train.max()


    def fit_curve(self) -> np.poly1d:
        """
        Ajusta un polinomio de grado `self.degree` a (vacancias_train, surface_area_train).
        Retorna el objeto np.poly1d.
        """
        if self.vacancias_train is None or self.surface_area_train is None:
            raise ValueError("Debes llamar a prepare_training_data() primero.")
        coef = np.polyfit(self.vacancias_train, self.surface_area_train, deg=self.degree)
        self.poly = np.poly1d(coef)
        return self.poly


    def predict_vacancies_from_area(
        self,
        observed_area: float,
        vacancy_range: tuple = (1, 9),
        area_range: tuple = (None, None)
    ) -> float:
        """
        Predice el número de vacancias para un área observada.
        - Si observed_area < min_area_train, retorna vacancy_range[0].
        - Si observed_area > max_area_train, retorna vacancy_range[1].
        - Sino, resuelve poly(x) = observed_area en el rango vacancy_range usando brentq.
        """
        if self.poly is None:
            raise ValueError("Ilímite: Debes llamar a fit_curve() primero.")

        min_area, max_area = area_range
        if min_area is not None and observed_area < min_area:
            return vacancy_range[0]
        if max_area is not None and observed_area > max_area:
            return vacancy_range[1]

        def f(x):
            return self.poly(x) - observed_area

        try:
            vac_pred = brentq(f, vacancy_range[0], vacancy_range[1])
            return vac_pred
        except ValueError:
            return None


    def plot_training_fit(self):
        """
        Genera un plot (sin guardarlo) de los puntos de entrenamiento contra la curva ajustada.
        (Puedes añadir código de Matplotlib aquí si lo deseas.)
        """
        if self.vacancias_train is None or self.surface_area_train is None:
            raise ValueError("Debes llamar a prepare_training_data() primero.")
        if self.poly is None:
            raise ValueError("Debes llamar a fit_curve() primero.")

        # Si quisieras generar un gráfico, acá iría algo como:
        # x_fit = np.linspace(self.vacancias_train.min(), self.vacancias_train.max(), 100)
        # y_fit = self.poly(x_fit)
        # plt.scatter(self.vacancias_train, self.surface_area_train, label='Datos')
        # plt.plot(x_fit, y_fit, label='Polinomio ajustado')
        # plt.legend()
        # plt.show()
        pass


    def predict_from_csv(self) -> pd.DataFrame:
        """
        Lee el CSV en `self.csv_path`, aplica `predict_vacancies_from_area`
        a cada valor de "area" y devuelve el DataFrame con una nueva columna
        "predicted_vacancies".
        """
        csv_data = pd.read_csv(self.csv_path)
        predictions = []
        for idx, row in csv_data.iterrows():
            observed_area = row["area"]
            pred = self.predict_vacancies_from_area(
                observed_area,
                vacancy_range=(1, 9),
                area_range=(self.min_area_train, self.max_area_train)
            )
            predictions.append(pred)
        csv_data["predicted_vacancies"] = predictions
        return csv_data
