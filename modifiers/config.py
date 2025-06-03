# modifiers/config.py

import os
import json

_def_ruta = os.path.join(os.path.dirname(__file__), "input_params.json")
if not os.path.exists(_def_ruta):
    raise FileNotFoundError(f"No se encontró 'input_params.json' en {os.path.dirname(__file__)}")

with open(_def_ruta, "r", encoding="utf-8") as f:
    _params = json.load(f)

if "CONFIG" not in _params or not isinstance(_params["CONFIG"], list) or len(_params["CONFIG"]) == 0:
    raise KeyError("El JSON debe contener 'CONFIG' como lista no vacía en input_params.json")

CONFIG = _params["CONFIG"]
PREDICTOR_COLUMNS = _params.get("PREDICTOR_COLUMNS", [])
