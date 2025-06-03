# main.py

import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

import ovito._extensions.pyscript
# … resto de imports de OVITO …

import os
from surface_processor.surface_processor import SurfaceProcessor
from surface_processor.cluster_dump_processor import ClusterDumpProcessor
from cluster_processing.cluster_processor import ClusterProcessor, ClusterProcessorMachine
from cluster_processing.key_files_separator import KeyFilesSeparator
from cluster_processing.export_cluster_list import ExportClusterList
import os
import json
from training.training_processor import TrainingProcessor
from training.vacancy_predictors import (
    VacancyPredictorRF,
    XGBoostVacancyPredictor,
    VacancyPredictor,
    VacancyPredictorMLP
)
from training.utils import load_json_data, resolve_input_params_path

import json
if __name__ == "__main__":
#training analysis

    # 1) Primero: correr el TrainingProcessor para generar training_data.json, key_single_vacancy.json, etc.
    processor = TrainingProcessor()
    processor.run()
    print("Entrenamiento completado. JSONs generados en 'outputs.vfinder/'.")


 
   
   






    try:
        with open("modifiers/input_params.json", "r", encoding="utf-8") as f:
            all_params = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de parámetros")
# … resto de extracción de CONFIG …
    # 2. Separar archivos críticos y finales
# 1. Ejecutar ClusterProcessor para generar key_areas.dump
    configuracion = all_params["CONFIG"][0]
    defect_file = configuracion['defect']
    processor = ClusterProcessor(defect_file)
    processor.run()
    separator = KeyFilesSeparator(configuracion, os.path.join("outputs.json", "clusters.json"))
    separator.run()

    # 3. Procesar dumps críticos (ClusterDumpProcessor)
    clave_criticos = ClusterDumpProcessor.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
    for archivo in clave_criticos:
        try:
            dump_proc = ClusterDumpProcessor(archivo, decimals=5)
            dump_proc.load_data()
            dump_proc.process_clusters()
            dump_proc.export_updated_file(f"{archivo}_actualizado.txt")
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")

    # 4. Reprocesar con ClusterProcessorMachine (subdivisión iterativa)
    lista_criticos = ClusterDumpProcessor.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
    for archivo in lista_criticos:
        machine_proc = ClusterProcessorMachine(archivo, configuracion['cluster tolerance'], configuracion['iteraciones_clusterig'])
        machine_proc.process_clusters()
        machine_proc.export_updated_file()

    # 5. Volver a separar archivos finales vs críticos
    separator = KeyFilesSeparator(configuracion, os.path.join("outputs.json", "clusters.json"))
    separator.run()

    # 6. Generar nuevos dumps por cluster (ExportClusterList)
    export_list = ExportClusterList("outputs.json/key_archivos.json")
    export_list.process_files()

    # 7. Calcular superficies de dump (SurfaceProcessor)
    surf_proc = SurfaceProcessor()
    surf_proc.process_all_files()
    surf_proc.export_results()

  


   # En lugar de resolve_input_params_path(...) simplemente:
    json_params = os.path.join(os.path.dirname(__file__), "input_params.json")
    with open(json_params, "r", encoding="utf-8") as f:
        params = json.load(f)

    # Ahora `params` contiene todo input_params.json  
    predictor_cols = params.get("PREDICTOR_COLUMNS", None)
    if predictor_cols is None:
        raise KeyError("input_params.json debe contener 'PREDICTOR_COLUMNS'.")

    # 3) Instanciar y probar los distintos predictivos:
    #    a) RandomForest
    rf_predictor = VacancyPredictorRF(
        json_path="outputs.vfinder/training_data.json",
        predictor_columns=predictor_cols
    )
    example_input = {col: 1.23 for col in predictor_cols}  # Ejemplo de diccionario de entrada
    vac_pred_rf = rf_predictor.predict_vacancies(**example_input)
    print("Predicción RF (vacancias):", vac_pred_rf)

    #    b) XGBoost
    xgb_predictor = XGBoostVacancyPredictor(
        training_data_path="outputs.vfinder/training_data.json",
        model_path="outputs.json/xgboost_model.json",
        predictor_columns=predictor_cols
    )
    # Para XGBoost, debes pasar una lista 2D de features:
    sample_features = [[example_input[col] for col in predictor_cols]]
    vac_pred_xgb = xgb_predictor.predict(sample_features)
    print("Predicción XGBoost (vacancias):", vac_pred_xgb)

   