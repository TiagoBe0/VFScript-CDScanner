
# modifiers/predictors/xgboost.py

import os
import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
import xgboost as xgb

class XGBoostVacancyPredictor:
    """
    XGBoost Regressor con cross‐validation y guardado de modelo.
    """
    def __init__(
        self,
        training_data_path: str = "outputs.vfinder/training_data.json",
        model_path: str = "outputs.json/xgboost_model.json",
        predictor_columns: list = None,
        n_splits: int = 5,
        random_state: int = 42
    ):
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.n_splits = n_splits
        self.random_state = random_state
        if predictor_columns is None:
            raise ValueError("Debes pasar predictor_columns explícitamente.")
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

        n_samples = X.shape[0]
        n_splits = (
            self.n_splits 
            if n_samples >= self.n_splits 
            else max(n_samples, 2)
        )
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, X, y, scoring='neg_mean_squared_error', cv=kfold)
        mse_scores = -scores  # Convertir a positivos

        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save_model(self.model_path)

    def predict(self, sample_input: np.ndarray) -> np.ndarray:
        sample_input = np.array(sample_input)
        sample_input = self.scaler.transform(sample_input)
        return self.model.predict(sample_input)
