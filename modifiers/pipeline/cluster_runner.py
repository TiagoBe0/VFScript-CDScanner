# modifiers/runner/vacancy_runner.py

import os
import math
import numpy as np
import pandas as pd
import json

class VacancyPredictionRunner:
    """
    Orquesta la predicción de vacancias usando:
      - Modelo lineal (predictor_small, predictor_large)
      - Random Forest (predictor_rf_small, predictor_rf_large) [si other_method=True]
      - XGBoost (predictor_xgb_small, predictor_xgb_large)
      - MLP (predictor_mlp_small, predictor_mlp_large) [si disponibles]

    Luego exporta totales, predicciones por cluster y acumuladas.
    """
    def __init__(
        self,
        archivo: str,
        predictor_small,
        predictor_large,
        predictor_rf_small,
        predictor_rf_large,
        predictor_xgb_small,
        predictor_xgb_large,
        predictor_mlp_small,
        predictor_mlp_large,
        other_method: bool = False,
        save_training: bool = False
    ):
        self.archivo = archivo

        # Umbrales de key_single_vacancy.json
        with open("outputs.vfinder/key_single_vacancy.json", "r", encoding="utf-8") as f:
            single_vac = json.load(f)
        self.ref_area = single_vac["surface_area"][0]
        self.ref_filled_volume = single_vac["filled_volume"][0]
        self.ref_vecinos = single_vac["cluster_size"][0]

        # Umbrales de key_double_vacancy.json
        with open("outputs.vfinder/key_double_vacancy.json", "r", encoding="utf-8") as f:
            diva_vac = json.load(f)
        self.ref_area_diva = diva_vac["surface_area"][0]
        self.ref_filled_volume_diva = diva_vac["filled_volume"][0]
        self.ref_vecinos_diva = diva_vac["cluster_size"][0]
        self.ref_mean_distance = diva_vac["mean_distance"][0]

        # Predictors
        self.predictor_small = predictor_small
        self.predictor_large = predictor_large
        self.predictor_rf_small = predictor_rf_small
        self.predictor_rf_large = predictor_rf_large
        self.predictor_xgb_small = predictor_xgb_small
        self.predictor_xgb_large = predictor_xgb_large
        self.predictor_mlp_small = predictor_mlp_small
        self.predictor_mlp_large = predictor_mlp_large

        self.other_method = other_method
        self.save_training = save_training

        self.vector_area = None
        self.vector_filled_volume = None
        self.vector_num_atm = None
        self.vector_true = None
        self.vector_mean_distance = None
        self.results = {}

    def load_data(self):
        df = pd.read_csv("outputs.json/resultados_procesados.csv")
        # Columnas: "archivo,mejor_radio,area,filled_volume,num_atm,mean_distance"
        self.vector_area = df["area"].values
        self.vector_num_atm = df["num_atm"].values
        self.vector_filled_volume = df["filled_volume"].values
        self.vector_mean_distance = df["mean_distance"].values if "mean_distance" in df.columns else None
        if "vacancys" in df.columns:
            self.vector_true = df["vacancys"].values
        else:
            self.vector_true = None

    def predict_linear(self) -> tuple:
        total_count = 0
        predictions = []
        errors = []
        threshold = 4 * self.ref_area

        for i, (area, filled_volume, num_atm, mean_distance) in enumerate(zip(
            self.vector_area,
            self.vector_filled_volume,
            self.vector_num_atm,
            self.vector_mean_distance
        )):
            # Condición directa → vacancia = 1
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"[Linear] Cluster {i}: Condición directa → Predicción 1")

            # Condición secundaria → vacancia = 2
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"[Linear] Cluster {i}: Condición secundaria → Predicción 2")

            else:
                features = {}
                if "surface_area" in self.predictor_small.columns:
                    features["surface_area"] = area
                if "filled_volume" in self.predictor_small.columns:
                    features["filled_volume"] = filled_volume
                if "cluster_size" in self.predictor_small.columns:
                    features["cluster_size"] = num_atm
                if "mean_distance" in self.predictor_small.columns:
                    features["mean_distance"] = mean_distance

                if area < threshold:
                    vacancias_pred = self.predictor_small.predict_vacancies(**features)
                    print(f"[Linear] Cluster {i}: Usando predictor_small → {vacancias_pred}")
                else:
                    vacancias_pred = self.predictor_large.predict_vacancies(**features)
                    print(f"[Linear] Cluster {i}: Usando predictor_large → {vacancias_pred}")

                total_count += vacancias_pred

            predictions.append(abs(vacancias_pred))
            if self.vector_true is not None:
                errors.append((abs(vacancias_pred) - self.vector_true[i]) ** 2)
            else:
                errors.append(None)

        print(f"[Linear] Total vacancias predichas = {abs(total_count)}\n")
        return abs(total_count), predictions, errors

    def predict_rf(self) -> tuple:
        if not self.other_method or self.predictor_rf_small is None or self.predictor_rf_large is None:
            return None, [], []
        total_count = 0
        predictions = []
        errors = []
        threshold = 4 * self.ref_area

        for i, (area, filled_volume, num_atm, mean_distance) in enumerate(zip(
            self.vector_area,
            self.vector_filled_volume,
            self.vector_num_atm,
            self.vector_mean_distance
        )):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"[RF] Cluster {i}: Condición directa → 1")

            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"[RF] Cluster {i}: Condición secundaria → 2")

            else:
                features = {}
                if "surface_area" in self.predictor_rf_small.columns:
                    features["surface_area"] = area
                if "filled_volume" in self.predictor_rf_small.columns:
                    features["filled_volume"] = filled_volume
                if "cluster_size" in self.predictor_rf_small.columns:
                    features["cluster_size"] = num_atm
                if "mean_distance" in self.predictor_rf_small.columns:
                    features["mean_distance"] = mean_distance

                if area < threshold:
                    vacancias_pred = self.predictor_rf_small.predict_vacancies(**features)
                    print(f"[RF] Cluster {i}: Usando predictor_rf_small → {vacancias_pred}")
                else:
                    vacancias_pred = self.predictor_rf_large.predict_vacancies(**features)
                    print(f"[RF] Cluster {i}: Usando predictor_rf_large → {vacancias_pred}")

                total_count += vacancias_pred

            predictions.append(abs(vacancias_pred))
            if self.vector_true is not None:
                errors.append((abs(vacancias_pred) - self.vector_true[i]) ** 2)
            else:
                errors.append(None)

        print(f"[RF] Total vacancias predichas = {abs(total_count)}\n")
        return abs(total_count), predictions, errors

    def predict_xgb(self) -> tuple:
        total_count = 0
        predictions = []
        errors = []
        threshold = 4 * self.ref_area

        for i, (area, filled_volume, num_atm, mean_distance) in enumerate(zip(
            self.vector_area,
            self.vector_filled_volume,
            self.vector_num_atm,
            self.vector_mean_distance
        )):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"[XGB] Cluster {i}: Condición directa → 1")

            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"[XGB] Cluster {i}: Condición secundaria → 2")

            else:
                features = {}
                if "surface_area" in self.predictor_xgb_small.columns:
                    features["surface_area"] = area
                if "filled_volume" in self.predictor_xgb_small.columns:
                    features["filled_volume"] = filled_volume
                if "cluster_size" in self.predictor_xgb_small.columns:
                    features["cluster_size"] = num_atm
                if "mean_distance" in self.predictor_xgb_small.columns:
                    features["mean_distance"] = mean_distance

                arr_input_small = np.array([[features[c] for c in self.predictor_xgb_small.columns]])
                arr_input_large = np.array([[features[c] for c in self.predictor_xgb_large.columns]])

                if area < threshold:
                    vacancias_pred = self.predictor_xgb_small.predict(arr_input_small)[0]
                    print(f"[XGB] Cluster {i}: Usando predictor_xgb_small → {vacancias_pred}")
                else:
                    vacancias_pred = self.predictor_xgb_large.predict(arr_input_large)[0]
                    print(f"[XGB] Cluster {i}: Usando predictor_xgb_large → {vacancias_pred}")

                total_count += vacancias_pred

            predictions.append(abs(vacancias_pred))
            if self.vector_true is not None:
                errors.append((abs(vacancias_pred) - self.vector_true[i]) ** 2)
            else:
                errors.append(None)

        print(f"[XGB] Total vacancias predichas = {abs(total_count)}\n")
        return abs(total_count), predictions, errors

    def predict_mlp(self) -> tuple:
        total_count = 0
        predictions = []
        errors = []
        threshold = 4 * self.ref_area

        for i, (area, filled_volume, num_atm, mean_distance) in enumerate(zip(
            self.vector_area,
            self.vector_filled_volume,
            self.vector_num_atm,
            self.vector_mean_distance
        )):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"[MLP] Cluster {i}: Condición directa → 1")

            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"[MLP] Cluster {i}: Condición secundaria → 2")

            else:
                features = {}
                if "surface_area" in self.predictor_mlp_small.columns:
                    features["surface_area"] = area
                if "filled_volume" in self.predictor_mlp_small.columns:
                    features["filled_volume"] = filled_volume
                if "cluster_size" in self.predictor_mlp_small.columns:
                    features["cluster_size"] = num_atm
                if "mean_distance" in self.predictor_mlp_small.columns:
                    features["mean_distance"] = mean_distance

                if area < threshold:
                    vacancias_pred = self.predictor_mlp_small.predict_vacancies(**features)
                    print(f"[MLP] Cluster {i}: Usando predictor_mlp_small → {vacancias_pred}")
                else:
                    vacancias_pred = self.predictor_mlp_large.predict_vacancies(**features)
                    print(f"[MLP] Cluster {i}: Usando predictor_mlp_large → {vacancias_pred}")

                total_count += vacancias_pred

            predictions.append(abs(vacancias_pred))
            if self.vector_true is not None:
                errors.append((abs(vacancias_pred) - self.vector_true[i]) ** 2)
            else:
                errors.append(None)

        print(f"[MLP] Total vacancias predichas = {abs(total_count)}\n")
        return abs(total_count), predictions, errors

    def run(self) -> dict:
        self.load_data()
        self.results = {}

        total_linear, pred_linear, err_linear = self.predict_linear()
        self.results["linear"] = {
            "total": total_linear,
            "predictions": pred_linear,
            "errors": err_linear
        }

        if self.predictor_mlp_small is not None and self.predictor_mlp_large is not None:
            total_mlp, pred_mlp, err_mlp = self.predict_mlp()
            self.results["mlp"] = {
                "total": total_mlp,
                "predictions": pred_mlp,
                "errors": err_mlp
            }

        if self.other_method:
            total_rf, pred_rf, err_rf = self.predict_rf()
            self.results["rf"] = {
                "total": total_rf,
                "predictions": pred_rf,
                "errors": err_rf
            }

        total_xgb, pred_xgb, err_xgb = self.predict_xgb()
        self.results["xgb"] = {
            "total": total_xgb,
            "predictions": pred_xgb,
            "errors": err_xgb
        }

        print("[Runner] Resumen de predicciones:")
        for method, data in self.results.items():
            print(f"  Modelo {method}: Total predicho = {data['total']}")

        return self.results

    def export_totals(self):
        """
        Exporta CSV con totales predichos para cada modelo.
        """
        totals_dict = {method: abs(data["total"]) for method, data in self.results.items()}
        df_totals = pd.DataFrame(list(totals_dict.items()), columns=["modelo", "contador_total"])
        output_dir = "outputs_tt"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(self.archivo)
        output_file = os.path.join(output_dir, f"{base_name}_totals.csv")
        df_totals.to_csv(output_file, index=False)
        print(f"[Runner] Archivo de totales exportado: {output_file}")

    def export_predictions_per_cluster(self, output_csv: str = None):
        """
        Exporta CSV con predicciones por cluster en la iteración actual.
        """
        n = len(self.results["linear"]["predictions"])
        df = pd.DataFrame({
            "cluster_id": list(range(n)),
            "linear": self.results["linear"]["predictions"],
            "rf": self.results["rf"]["predictions"] if "rf" in self.results else [None] * n,
            "xgb": self.results["xgb"]["predictions"],
            "mlp": self.results["mlp"]["predictions"] if "mlp" in self.results else [None] * n,
        })
        if self.vector_true is not None:
            df["true_vacancys"] = self.vector_true

        if output_csv is None:
            base_name = os.path.basename(self.archivo)
            output_csv = os.path.join("outputs_tt", f"{base_name}_predictions_per_cluster.csv")

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"[Runner] Archivo de predicciones por cluster exportado: {output_csv}")

    def export_predictions_accumulated(self, iteration: int, output_csv: str = "outputs_tt/accumulated_predictions.csv"):
        """
        Exporta o acumula las predicciones de la iteración actual en un CSV global.
        """
        n = len(self.results["linear"]["predictions"])
        df_iteration = pd.DataFrame({
            "iteration": [iteration] * n,
            "cluster_id": list(range(n)),
            "linear": self.results["linear"]["predictions"],
            "rf": self.results["rf"]["predictions"] if "rf" in self.results else [None] * n,
            "xgb": self.results["xgb"]["predictions"],
            "mlp": self.results["mlp"]["predictions"] if "mlp" in self.results else [None] * n,
        })
        if self.vector_true is not None:
            df_iteration["true_vacancys"] = self.vector_true

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        if os.path.exists(output_csv):
            df_iteration.to_csv(output_csv, mode='a', index=False, header=False)
        else:
            df_iteration.to_csv(output_csv, index=False)
        print(f"[Runner] Archivo acumulado exportado: {output_csv}")
