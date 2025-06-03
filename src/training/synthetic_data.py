# modifiers/training/synthetic_data.py

import json
import numpy as np
from scipy.interpolate import interp1d
import os
class SyntheticDataGenerator:
    def __init__(self, data: dict, num_points: int = 100, interpolation_kind: str = 'linear'):
        """
        data: diccionario con claves:
          ["surface_area", "filled_volume", "vacancys", "cluster_size", "mean_distance"]
        num_points: cuántos puntos sintéticos generar
        interpolation_kind: tipo de interpolación (‘linear’, ‘quadratic’, etc.)
        """
        self.data = data
        self.num_points = num_points
        self.interpolation_kind = interpolation_kind

        required_keys = ["surface_area", "filled_volume", "vacancys", "cluster_size", "mean_distance"]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"La clave '{key}' no está en los datos de entrada.")

        self.vacancias = np.array(self.data["vacancys"])


    def generate(self) -> dict:
        """
        Genera un diccionario con datos sintéticos de tamaño self.num_points,
        interpolando cada serie en `self.data` sobre el rango de vacancias.
        """
        vac_new = np.linspace(self.vacancias.min(), self.vacancias.max(), self.num_points)

        # Interpoladores
        interp_sm = interp1d(self.vacancias, self.data["surface_area"], kind=self.interpolation_kind)
        sm_new = interp_sm(vac_new)

        interp_filled = interp1d(self.vacancias, self.data["filled_volume"], kind=self.interpolation_kind)
        filled_new = interp_filled(vac_new)

        interp_vecinos = interp1d(self.vacancias, self.data["cluster_size"], kind=self.interpolation_kind)
        vecinos_new = np.round(interp_vecinos(vac_new)).astype(int)

        interp_mean = interp1d(self.vacancias, self.data["mean_distance"], kind=self.interpolation_kind)
        mean_new = np.round(interp_mean(vac_new)).astype(int)

        synthetic_data = {
            "surface_area":   sm_new.tolist(),
            "filled_volume":  filled_new.tolist(),
            "vacancys":       vac_new.tolist(),
            "cluster_size":   vecinos_new.tolist(),
            "mean_distance":  mean_new.tolist()
        }
        return synthetic_data


    def export_to_json(self, output_path: str, data: dict):
        """
        Escribe `data` en un JSON en `output_path`.
        """
        dirname = os.path.dirname(output_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        print(f"Datos exportados a {output_path}")
