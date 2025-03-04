import os
import json
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, ClusterAnalysisModifier, ConstructSurfaceModifier, InvertSelectionModifier
from input_params import CONFIG, PREDICTOR_COLUMNS
from DefectAnalysisStepScript import *
from DefectAnalysisStepScript import CDScanner
from TrainingStepScript import *
import logging

class VacancyPredictionRunner:
    def __init__(self, archivo, predictor_small, predictor_large, predictor_rf_small, predictor_rf_large, predictor_xgb_small, predictor_xgb_large,predictor_mlp_small , predictor_mlp_large, other_method=False, save_training=False):
        self.archivo = archivo
        with open("outputs.vfinder/key_single_vacancy.json", "r") as f:
            single_vac = json.load(f)
        self.ref_area = single_vac["surface_area"][0]
        self.ref_filled_volume = single_vac["filled_volume"][0]
        self.ref_vecinos = single_vac["cluster_size"][0]
        with open("outputs.vfinder/key_double_vacancy.json", "r") as f:
            diva_vac = json.load(f)
        self.ref_area_diva = diva_vac["surface_area"][0]
        self.ref_filled_volume_diva = diva_vac["filled_volume"][0]
        self.ref_vecinos_diva = diva_vac["cluster_size"][0]
        self.ref_mean_distance = diva_vac["mean_distance"][0]
        self.predictor_small = predictor_small
        self.predictor_large = predictor_large
        self.predictor_rf_small = predictor_rf_small
        self.predictor_rf_large = predictor_rf_large
        self.predictor_xgb_small = predictor_xgb_small
        self.predictor_xgb_large = predictor_xgb_large
        self.predictor_mlp_small=predictor_mlp_small
        self.predictor_mlp_large = predictor_mlp_large
        self.other_method = other_method
        self.save_training = save_training
        self.vector_area = None
        self.vector_filled_volume = None
        self.vector_num_atm = None
        self.vector_true = None
        self.vector_mean_distance = None
        self.results = {}
        self.COLUMNS_TRAINING = PREDICTOR_COLUMNS

    def load_data(self):
        df = pd.read_csv("outputs.json/resultados_procesados.csv")
        self.vector_area = df["area"].values
        self.vector_num_atm = df["num_atm"].values
        self.vector_filled_volume = df["filled_volume"].values
        self.vector_mean_distance = df["mean_distance"].values if "mean_distance" in df.columns else None
        if "vacancys" in df.columns:
            self.vector_true = df["vacancys"].values
        else:
            self.vector_true = None

    def predict_linear(self):
        total_count = 0
        predictions = []
        errors = []
        threshold = 4* self.ref_area
        for i, (area, filled_volume, num_atm, mean_distance) in enumerate(zip(self.vector_area, self.vector_filled_volume, self.vector_num_atm, self.vector_mean_distance)):
            
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"[Linear] Cluster {i}: Condición directa. Predicción: {vacancias_pred}. Características: area={area}, filled_volume={filled_volume}, num_atm={num_atm}, mean_distance={mean_distance}")
            
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or 
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or 
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"[Linear] Cluster {i}: Condición secundaria. Predicción: {vacancias_pred}. Características: area={area}, filled_volume={filled_volume}, num_atm={num_atm}, mean_distance={mean_distance}")
            
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
                    print(f"[Linear] Cluster {i}: Usando predictor_small con features {features}. Predicción: {vacancias_pred}")
                else:
                    vacancias_pred = self.predictor_large.predict_vacancies(**features)
                    print(f"[Linear] Cluster {i}: Usando predictor_large con features {features}. Predicción: {vacancias_pred}")
                total_count += vacancias_pred
            predictions.append(abs(vacancias_pred))
            if self.vector_true is not None:
                error = (abs(vacancias_pred) - self.vector_true[i]) ** 2
                errors.append(error)
            else:
                errors.append(None)
        print(f"[Linear] Total de vacancias predichas: {abs(total_count)}\n")
        return abs(total_count), predictions, errors

    def predict_rf(self):
        if not self.other_method or self.predictor_rf_small is None or self.predictor_rf_large is None:
            return None, [], []
        total_count = 0
        predictions = []
        errors = []
        threshold = 4* self.ref_area
        for i, (area, filled_volume, num_atm, mean_distance) in enumerate(zip(self.vector_area, self.vector_filled_volume, self.vector_num_atm, self.vector_mean_distance)):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"[RF] Cluster {i}: Condición directa. Predicción: {vacancias_pred}. Características: area={area}, filled_volume={filled_volume}, num_atm={num_atm}, mean_distance={mean_distance}")
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or 
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or 
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"[RF] Cluster {i}: Condición secundaria. Predicción: {vacancias_pred}. Características: area={area}, filled_volume={filled_volume}, num_atm={num_atm}, mean_distance={mean_distance}")
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
                    print(f"[RF] Cluster {i}: Usando predictor_rf_small con features {features}. Predicción: {vacancias_pred}")
                else:
                    vacancias_pred = self.predictor_rf_large.predict_vacancies(**features)
                    print(f"[RF] Cluster {i}: Usando predictor_rf_large con features {features}. Predicción: {vacancias_pred}")
                total_count += vacancias_pred
            predictions.append(abs(vacancias_pred))
            if self.vector_true is not None:
                error = (abs(vacancias_pred) - self.vector_true[i]) ** 2
                errors.append(error)
            else:
                errors.append(None)
        print(f"[RF] Total de vacancias predichas: {abs(total_count)}\n")
        return abs(total_count), predictions, errors

    def predict_xgb(self):
        total_count = 0
        predictions = []
        errors = []
        threshold = 4* self.ref_area
        for i, (area, filled_volume, num_atm, mean_distance) in enumerate(zip(self.vector_area, self.vector_filled_volume, self.vector_num_atm, self.vector_mean_distance)):
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"[XGB] Cluster {i}: Condición directa. Predicción: {vacancias_pred}. Características: area={area}, filled_volume={filled_volume}, num_atm={num_atm}, mean_distance={mean_distance}")
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or 
                  math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or 
                  (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"[XGB] Cluster {i}: Condición secundaria. Predicción: {vacancias_pred}. Características: area={area}, filled_volume={filled_volume}, num_atm={num_atm}, mean_distance={mean_distance}")
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
                if area < threshold:
                    sample_input = np.array([[features[col] for col in self.predictor_xgb_small.columns]])
                    vacancias_pred = self.predictor_xgb_small.predict(sample_input)[0]
                    print(f"[XGB] Cluster {i}: Usando predictor_xgb_small con features {features}. Predicción: {vacancias_pred}")
                else:
                    sample_input = np.array([[features[col] for col in self.predictor_xgb_large.columns]])
                    vacancias_pred = self.predictor_xgb_large.predict(sample_input)[0]
                    print(f"[XGB] Cluster {i}: Usando predictor_xgb_large con features {features}. Predicción: {vacancias_pred}")
                total_count += vacancias_pred
            predictions.append(abs(vacancias_pred))
            if self.vector_true is not None:
                error = (abs(vacancias_pred) - self.vector_true[i]) ** 2
                errors.append(error)
            else:
                errors.append(None)
        print(f"[XGB] Total de vacancias predichas: {abs(total_count)}\n")
        return abs(total_count), predictions, errors
    def predict_mlp(self):
        total_count = 0
        predictions = []
        errors = []
        threshold =4* self.ref_area
        for i, (area, filled_volume, num_atm, mean_distance) in enumerate(zip(self.vector_area, self.vector_filled_volume, self.vector_num_atm, self.vector_mean_distance)):
            # Condición directa
            if (math.isclose(area, self.ref_area, rel_tol=0.3) or 
                math.isclose(filled_volume, self.ref_filled_volume, rel_tol=0.3) or 
                (num_atm == self.ref_vecinos)):
                vacancias_pred = 1
                total_count += 1
                print(f"[MLP] Cluster {i}: Condición directa. Predicción: {vacancias_pred}. Características: area={area}, filled_volume={filled_volume}, num_atm={num_atm}, mean_distance={mean_distance}")
            # Condición secundaria
            elif (math.isclose(area, self.ref_area_diva, rel_tol=0.2) or 
                math.isclose(filled_volume, self.ref_filled_volume_diva, rel_tol=0.2) or 
                (num_atm == self.ref_vecinos_diva)):
                vacancias_pred = 2
                total_count += 2
                print(f"[MLP] Cluster {i}: Condición secundaria. Predicción: {vacancias_pred}. Características: area={area}, filled_volume={filled_volume}, num_atm={num_atm}, mean_distance={mean_distance}")
            # Uso del modelo MLP
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
                    print(f"[MLP] Cluster {i}: Usando predictor_mlp_small con features {features}. Predicción: {vacancias_pred}")
                else:
                    vacancias_pred = self.predictor_mlp_large.predict_vacancies(**features)
                    print(f"[MLP] Cluster {i}: Usando predictor_mlp_large con features {features}. Predicción: {vacancias_pred}")
                total_count += vacancias_pred
            predictions.append(abs(vacancias_pred))
            if self.vector_true is not None:
                error = (abs(vacancias_pred) - self.vector_true[i]) ** 2
                errors.append(error)
            else:
                errors.append(None)
        print(f"[MLP] Total de vacancias predichas: {abs(total_count)}\n")
        return abs(total_count), predictions, errors

    def run(self):
        self.load_data()
        self.results = {}
        total_linear, pred_linear, err_linear = self.predict_linear()
        self.results["linear"] = {"total": total_linear, "predictions": pred_linear, "errors": err_linear}
        
        # Incluir el modelo MLP si está disponible
        if self.predictor_mlp_small is not None and self.predictor_mlp_large is not None:
            total_mlp, pred_mlp, err_mlp = self.predict_mlp()
            self.results["mlp"] = {"total": total_mlp, "predictions": pred_mlp, "errors": err_mlp}
        
        if self.other_method:
            total_rf, pred_rf, err_rf = self.predict_rf()
            self.results["rf"] = {"total": total_rf, "predictions": pred_rf, "errors": err_rf}
        total_xgb, pred_xgb, err_xgb = self.predict_xgb()
        self.results["xgb"] = {"total": total_xgb, "predictions": pred_xgb, "errors": err_xgb}
        
        print("Resumen de predicciones:")
        for method, data in self.results.items():
            print(f"  Modelo {method}: Total predicho = {data['total']}")
        return self.results

    def export_totals(self):
        """
        Exporta un CSV con el total de vacancias predichas por cada modelo.
        """
        totals_dict = {method: abs(data["total"]) for method, data in self.results.items()}
        df_totals = pd.DataFrame(list(totals_dict.items()), columns=["modelo", "contador_total"])
        output_dir = "outputs_tt"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(self.archivo)
        output_file = os.path.join(output_dir, f"{base_name}_totals.csv")
        df_totals.to_csv(output_file, index=False)
        print(f"Archivo de totales exportado: {output_file}")
    def export_predictions_per_cluster(self, output_csv=None):
        """
        Exporta un CSV con las predicciones de vacancias por cada cluster en la iteración actual.
        Cada fila contiene el id del cluster y la vacancia predicha por cada modelo.
        Se utiliza el nombre base del archivo de entrada para nombrar el CSV y evitar sobrescritura.
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

        # Si no se pasa un output_csv, se genera a partir del nombre base de self.archivo
        if output_csv is None:
            base_name = os.path.basename(self.archivo)
            output_csv = os.path.join("outputs_tt", f"{base_name}_predictions_per_cluster.csv")

        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Archivo de predicciones por cluster exportado: {output_csv}")


    def export_predictions_accumulated(self, iteration, output_csv="outputs_tt/accumulated_predictions.csv"):
        """
        Exporta (o acumula) las predicciones de la iteración actual en un CSV general.
        Si el archivo ya existe, se añaden las filas sin sobrescribir el contenido previo.
        Se agrega una columna 'iteration' para identificar la corrida.
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
        print(f"Archivo acumulado de predicciones exportado/actualizado: {output_csv}")


if __name__ == "__main__":
    # Configuración y procesamiento previo
    config = CONFIG[0]
    defect_file_list = config['defect']
    relax = config['relax']
    radius_training = config['radius_training']
    radius = config['radius']
    smoothing_level_training = config['smoothing_level_training']
    other_method = config['other method']
    save_training = config['save_training']
    strees = config['strees']
    if config['training_step']:
        processor = TrainingProcessor(relax, radius_training, radius, smoothing_level_training, strees, save_training)
        processor.run()

    ####################################################################################

    for file_defect in defect_file_list:
        processor = ClusterProcessor(file_defect)
        processor.run()
        config = CONFIG[0]
        separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
        separator.run()
        json_path = "outputs.json/key_archivos.json"
        
        separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
        separator.run()
        json_path = "outputs.json/key_archivos.json"
        archivos = ClusterDumpProcessor.cargar_lista_archivos_criticos(json_path)
        for archivo in archivos:
            try:
                processor = ClusterDumpProcessor(archivo, decimals=5)
                processor.load_data()
                processor.process_clusters()
                processor.export_updated_file(f"{archivo}")
            except Exception as e:
                print(f"Error procesando {archivo}: {e}")
        lista_criticos = UtilidadesClustering.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
        for archivo in lista_criticos:
            processor = ClusterProcessorMachine(archivo, threshold=config['cluster tolerance'], max_iterations=config['iteraciones_clusterig'])
            processor.process_clusters()
            processor.export_updated_file()
        config = CONFIG[0]
        separator = KeyFilesSeparator(config, os.path.join("outputs.json", "clusters.json"))
        separator.run()
        processor_0 = ExportClusterList("outputs.json/key_archivos.json")
        processor_0.process_files()
        processor_1 = SurfaceProcessor()
        processor_1.process_all_files()
        processor_1.export_results()

        ####################################################
        if config['generic_data']:
            json_file_path = "outputs.vfinder/training_data.json"  
            with open(json_file_path, "r") as f:
                data_original = json.load(f)
            generator = SyntheticDataGenerator(data_original, num_points=100, interpolation_kind='linear')
            synthetic_data = generator.generate()
            output_json_path = "outputs.vfinder/training_data.json"
            generator.export_to_json(output_json_path, synthetic_data)
        columns_training=PREDICTOR_COLUMNS
        predictor_small = VacancyPredictor("outputs.vfinder/training_data.json")
        predictor_large = VacancyPredictor("outputs.vfinder/training_data.json")

        predictor_rf_small = VacancyPredictorRF("outputs.vfinder/training_small.json")
        predictor_rf_large = VacancyPredictorRF("outputs.vfinder/training_data.json")

        predictor_xgb_small = XGBoostVacancyPredictor("outputs.vfinder/training_small.json")
        predictor_xgb_large = XGBoostVacancyPredictor("outputs.vfinder/training_data.json")


        predictor_mlp_small = VacancyPredictorMLP("outputs.vfinder/training_data.json")
        predictor_mlp_large = VacancyPredictorMLP("outputs.vfinder/training_data.json")

        runner = VacancyPredictionRunner(
            archivo=file_defect,
            predictor_small=predictor_small,
            predictor_large=predictor_large,
            predictor_rf_small=predictor_rf_small,
            predictor_rf_large=predictor_rf_large,
            predictor_xgb_small=predictor_xgb_small,
            predictor_xgb_large=predictor_xgb_large,
            predictor_mlp_small=predictor_mlp_small,
            predictor_mlp_large=predictor_mlp_large,
            other_method=True
        )
        runner.run()
        runner.export_totals()
        runner.export_predictions_per_cluster()  
        iteration = 1  
        runner.export_predictions_accumulated(iteration)
        
        predictor = VacancyPredictorCurve("outputs.vfinder/training_data.json", "outputs.json/resultados_procesados.csv", degree=3)
        predictor.load_training_data(as_dataframe=True)
        predictor.prepare_training_data()
        poly = predictor.fit_curve()
        csv_with_predictions = predictor.predict_from_csv()

        if config['chat']:
            from chat import CSVInterpreter
            interpreter = CSVInterpreter(f"outputs.vfinder/{file_defect}.csv")
            resultado = interpreter.interpret(prompt="Interpreta los siguientes datos:")
            print(resultado)


###############################3


        if config['CDScanner']:
                scanner = CDScanner()
                scanner.process_clusters()
                scanner.run_kmeans()
                scanner.create_dataframe()
                scanner.update_dump_header()
                scanner.export_dump_file()
                scanner.calculate_silhouette()
                scanner.export_mapping()
                scanner.export_figures()
        if config['Histograma']:
            
            ruta_archivo_csv = 'outputs.csv/mapeo_archivos_cluster.csv'
            archivo_csv = HistogramaArchivoCSV(ruta_archivo_csv)
            archivo_csv.leer_archivo()
            procesador_archivos = HistogramaFile(archivo_csv.nombres_archivos)
            procesador_archivos.procesar_archivos(archivo_csv.clusters)
            graficador = HistogramaTypes(procesador_archivos.resultados)
            tipos_unicos = graficador.obtener_tipos_unicos()
            clusters_unicos = sorted(set(archivo_csv.clusters))
            matriz_frecuencias = graficador.crear_matriz_frecuencias(clusters_unicos, tipos_unicos)
            graficador.graficar(matriz_frecuencias, tipos_unicos, ruta_archivo_csv)

       