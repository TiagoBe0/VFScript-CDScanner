# modifiers/training/utils.py

import os
import json
import pandas as pd

def load_json_data(json_path: str) -> pd.DataFrame:
    """
    Lee un JSON del disco y devuelve un DataFrame de pandas con su contenido.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def resolve_input_params_path(module_file: str, json_name: str = "input_params.json") -> str:
    """
    Dado el __file__ de un módulo, sube un nivel y
    busca 'input_params.json' en esa carpeta. Si no existe,
    lanza FileNotFoundError.
    """
    carpeta_actual = os.path.dirname(module_file)       # e.g. .../modifiers/training
    carpeta_raiz   = os.path.dirname(carpeta_actual)     # e.g. .../modifiers
    ruta_json = os.path.join(carpeta_raiz, json_name)
    if not os.path.exists(ruta_json):
        raise FileNotFoundError(f"No se encontró '{json_name}' en: {carpeta_raiz}")
    return ruta_json
