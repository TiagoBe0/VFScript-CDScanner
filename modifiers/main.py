# main.py

import warnings
warnings.filterwarnings('ignore', message='.*OVITO.*PyPI')

import ovito._extensions.pyscript
# … resto de imports de OVITO …
import pandas as pd
import os
from surface_processor.surface_processor import SurfaceProcessor
from surface_processor.cluster_dump_processor import ClusterDumpProcessor
from cluster_processing.cluster_processor import ClusterProcessor, ClusterProcessorMachine
from cluster_processing.key_files_separator import KeyFilesSeparator
from cluster_processing.export_cluster_list import ExportClusterList
from training.cristal_structure_gen import CrystalStructureGenerator
import os
import json
from cluster_processing.cluster_macth import DumpProcessorFinger, StatisticsCalculatorFinger, JSONFeatureExporterFinger
from training.training_processor import TrainingProcessor
from training.vacancy_predictors import (
    VacancyPredictorRF,
    XGBoostVacancyPredictor,
    VacancyPredictor,
    VacancyPredictorMLP
)

from pathlib import Path
from training.utils import load_json_data, resolve_input_params_path
from runner.finger_runner import WinnerFinger
import json



if __name__ == "__main__":
#training analysis

    # 0) Configurar directorios
    

    base = "outputs"
    for sub in ("csv", "dump", "json"):
        os.makedirs(os.path.join(base, sub), exist_ok=True)

    # 1) Primero: correr el TrainingProcessor para generar training_data.json, key_single_vacancy.json, etc.
    processor = TrainingProcessor()
    processor.run()


 





    try:
        with open("modifiers/input_params.json", "r", encoding="utf-8") as f:
            all_params = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"No se encontró el archivo de parámetros")
    
    configuracion = all_params["CONFIG"][0]
    defect_file = configuracion['defect']

    cs_out_dir = Path("inputs/dump")
    cs_generator = CrystalStructureGenerator(configuracion, cs_out_dir)
    dump_path = cs_generator.generate()
    print(f"Estructura relajada generada en: {dump_path}")

   






    processor = ClusterProcessor(defect_file)
    processor.run()
    separator = KeyFilesSeparator(configuracion, os.path.join("outputs/json", "clusters.json"))
    separator.run()

    # 3. Procesar dumps críticos (ClusterDumpProcessor)
    clave_criticos = ClusterDumpProcessor.cargar_lista_archivos_criticos("outputs/json/key_archivos.json")
    for archivo in clave_criticos:
        try:
            dump_proc = ClusterDumpProcessor(archivo, decimals=5)
            dump_proc.load_data()
            dump_proc.process_clusters()
            dump_proc.export_updated_file(f"{archivo}_actualizado.txt")
        except Exception as e:
            print(f"Error procesando {archivo}: {e}")

    # 4. Reprocesar con ClusterProcessorMachine (subdivisión iterativa)
    lista_criticos = ClusterDumpProcessor.cargar_lista_archivos_criticos("outputs/json/key_archivos.json")
    for archivo in lista_criticos:
        machine_proc = ClusterProcessorMachine(archivo, configuracion['cluster tolerance'], configuracion['iteraciones_clusterig'])
        machine_proc.process_clusters()
        machine_proc.export_updated_file()

    # 5. Volver a separar archivos finales vs críticos
    separator = KeyFilesSeparator(configuracion, os.path.join("outputs/json", "clusters.json"))
    separator.run()

    # 6. Generar nuevos dumps por cluster (ExportClusterList)
    export_list = ExportClusterList("outputs/json/key_archivos.json")
    export_list.process_files()

    # 7. Calcular superficies de dump (SurfaceProcessor)
    surf_proc = SurfaceProcessor(configuracion)
    surf_proc.process_all_files()
    surf_proc.export_results()


    # ———> BLOQUE DE PREDICCIÓN EXTRAÍDO DE CSV Y CON LA CLAVE EN EL RAÍZ DEL JSON
    json_params = os.path.join(os.path.dirname(__file__), "input_params.json")
    with open(json_params, "r", encoding="utf-8") as f:
        params = json.load(f)

    # Ahora obtenemos PREDICTOR_COLUMNS del nivel raíz, no dentro de CONFIG
    predictor_cols = params.get("PREDICTOR_COLUMNS", None)
    if predictor_cols is None or not isinstance(predictor_cols, list) or len(predictor_cols) == 0:
        raise KeyError("input_params.json debe contener 'PREDICTOR_COLUMNS' (lista no vacía) en el nivel raíz.")

    # 1) Instanciar los modelos predictivos (ya entrenados)
    rf_predictor = VacancyPredictorRF(
        json_path="outputs/json/training_data.json",
        predictor_columns=predictor_cols
    )
    xgb_predictor = XGBoostVacancyPredictor(
        training_data_path="outputs/json/training_data.json",
        model_path="outputs/json/xgboost_model.json",
        predictor_columns=predictor_cols
    )

    # 2) Leer el CSV con las features (ajusta la ruta según corresponda)
    csv_path = os.path.join("outputs", "csv", "defect_data.csv")
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No se encontró el CSV en: {csv_path}")

    df = pd.read_csv(csv_path)

    # 3) Iterar sobre cada fila del DataFrame y predecir
    print(f"\********** Predicciones para defecto {defect_file} usando {csv_path} *************")
    for idx, row in df.iterrows():
        # Construir el diccionario solo con las columnas que esperan los modelos
        try:
            features = { col: row[col] for col in predictor_cols }
        except KeyError as e:
            raise KeyError(
                f"La columna {e} no existe en el CSV. "
                f"Columnas disponibles: {list(df.columns)}"
            )

        # RandomForest
        vac_pred_rf = rf_predictor.predict_vacancies(**features)

        sample_features = [[features[col] for col in predictor_cols]]
        vac_pred_xgb = xgb_predictor.predict(sample_features)

        print(f"Fila {idx} → features: {features}")
        print(f"  • Predicción RF (vacancias): {vac_pred_rf}")
        print(f"  • Predicción XGBoost (vacancias): {vac_pred_xgb}\n")


        #fingerprints integration
        json_input      = "outputs/json/key_archivos.json"
        output_csv_path = "outputs/csv/finger_defect_data.csv"
        exporter = JSONFeatureExporterFinger(json_input, output_csv_path)
        exporter.export()
        winner_finder  = WinnerFinger(
            defect_csv=Path("outputs/csv/finger_defect_data.csv"),
            normal_csv=Path("outputs/csv/finger_data.csv"),
            output_csv=Path("outputs/csv/finger_winner_data.csv"),
            id_col="file_name"
        )
        winner_finder.run()

    # 1. Rutas a tus archivos
    defect_csv   = Path('outputs/csv/defect_data.csv')
    finger_csv   = Path('outputs/csv/finger_winner_data.csv')
    output_csv   = Path('outputs/csv/defect_data.csv')  # salida

    # 2. Carga de datos
    defect_df = pd.read_csv(defect_csv)
    finger_df = pd.read_csv(finger_csv)

    # 3. Extraer sólo las columnas clave de finger_df
    #    Asumimos que en defect_df la columna 'archivo' tiene rutas como ".../key_area_1.dump"
    #    y en finger_df la columna 'defect_file' sólo el nombre "key_area_1.dump".
    finger_df = finger_df[['defect_file', 'fingerprint']]

    # 4. Crear en defect_df una columna con sólo el nombre de archivo
    defect_df['file_name'] = defect_df['archivo'].apply(lambda p: Path(p).name)

    # 5. Hacer merge para añadir 'fingerprint'
    enriched = defect_df.merge(
        finger_df,
        how='left',
        left_on='file_name',
        right_on='defect_file'
    )

    # 6. Limpiar columnas auxiliares y reordenar si lo deseas
    enriched = enriched.drop(columns=['file_name', 'defect_file'])

    # 7. Guardar resultado
    enriched.to_csv(output_csv, index=False)
    print(f'CSV enriquecido guardado en: {output_csv}')
