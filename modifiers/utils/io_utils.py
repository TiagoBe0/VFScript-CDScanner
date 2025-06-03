# modifiers/utils/io_utils.py

import os
import json
import pandas as pd

def load_json_data(json_path: str) -> pd.DataFrame:
    """
    Lee un JSON y devuelve un DataFrame de pandas.
    """
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return pd.DataFrame(data)

def resolve_input_params_path(module_file: str, json_name: str = "input_params.json") -> str:
    """
    Busca primero 'input_params.json' en la misma carpeta que module_file.
    Si no existe, sube un nivel y busca allí.
    """
    carpeta_actual = os.path.dirname(module_file)
    ruta_local = os.path.join(carpeta_actual, json_name)
    if os.path.exists(ruta_local):
        return ruta_local

    carpeta_padre = os.path.dirname(carpeta_actual)
    ruta_padre = os.path.join(carpeta_padre, json_name)
    if os.path.exists(ruta_padre):
        return ruta_padre

    raise FileNotFoundError(f"No se encontró '{json_name}' en {carpeta_actual} ni en {carpeta_padre}")
