import os
import json
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier,DislocationAnalysisModifier, DeleteSelectedModifier, ClusterAnalysisModifier, ConstructSurfaceModifier, InvertSelectionModifier
from DefectAnalysisStepScript import *
from DefectAnalysisStepScript import CDScanner
from TrainingStepScript import *
import logging
#necesito dejar solo el modelo xgboost para predicciones los demas se deben borrar (Ilineal,mlp,randomforest)
class VacancyPredictionRunner:
    def __init__(self, archivo, predictor_xgb_small, predictor_xgb_large, other_method=False, save_training=False):
        self.archivo = archivo
        with open("outputs.vfinder/key_single_vacancy.json", "r") as f:
            single_vac = json.load(f)
        self.ref_area = single_vac["surface_area"][0]
        self.ref_filled_volume = single_vac["filled_volume"][0]
        self.ref_vecinos = single_vac["cluster_size"][0]
        self.ref_mean_distance_single = single_vac["mean_distance"][0]
        with open("outputs.vfinder/key_double_vacancy.json", "r") as f:
            diva_vac = json.load(f)
        self.ref_area_diva = diva_vac["surface_area"][0]
        self.ref_filled_volume_diva = diva_vac["filled_volume"][0]
        self.ref_vecinos_diva = diva_vac["cluster_size"][0]
        self.ref_mean_distance_diva = diva_vac["mean_distance"][0]
        with open("outputs.vfinder/key_tri_vacancy.json", "r") as f:
            tri_vac = json.load(f)
        self.ref_area_tri = tri_vac["surface_area"][0]
        self.ref_filled_volume_tri = tri_vac["filled_volume"][0]
        self.ref_vecinos_tri = tri_vac["cluster_size"][0]
        self.ref_mean_distance_tri = tri_vac["mean_distance"][0]
        with open("outputs.vfinder/key_cua_vacancy.json", "r") as f:
            cua_vac = json.load(f)
        self.ref_area_cua = cua_vac["surface_area"][0]
        self.ref_filled_volume_cua = cua_vac["filled_volume"][0]
        self.ref_vecinos_cua = cua_vac["cluster_size"][0]
        self.ref_mean_distance_cua = cua_vac["mean_distance"][0]
        with open("outputs.vfinder/key_cin_vacancy.json", "r") as f:
            cin_vac = json.load(f)
        self.ref_area_cin = cin_vac["surface_area"][0]
        self.ref_filled_volume_cin = cin_vac["filled_volume"][0]
        self.ref_vecinos_cin = cin_vac["cluster_size"][0]
        self.ref_mean_distance_cin = cin_vac["mean_distance"][0]
        with open("outputs.vfinder/key_six_vacancy.json", "r") as f:
            six_vac = json.load(f)
        self.ref_area_six = six_vac["surface_area"][0]
        self.ref_filled_volume_six = six_vac["filled_volume"][0]
        self.ref_vecinos_six = six_vac["cluster_size"][0]
        self.ref_mean_distance_six = six_vac["mean_distance"][0]
        self.predictor_xgb_small = predictor_xgb_small
        self.predictor_xgb_large = predictor_xgb_large
        self.predictor_rf_force=1
        self.other_method = other_method
        self.save_training = save_training
        self.ref_area_cin_rf = six_vac["surface_area"][0]
        self.ref_filled_volume_cin_rf =six_vac["filled_volume"][0]
        self.ref_vecinos_cin_rf = six_vac["cluster_size"][0]
        self.ref_mean_distance_cin_rf =six_vac["mean_distance"][0]
        self.vector_area = None
        self.vector_filled_volume = None
        self.vector_num_atm = None
        self.vector_true = None
        self.vector_mean_distance = None
        self.results = {}
        with open('input_params.json', 'r') as file:
            data = json.load(file)

        print(type(data))  # Asegurate que sea <class 'dict'>

        # Si es dict, accedé a la clave
        if isinstance(data, dict):
            predictor_columns = data.get("PREDICTOR_COLUMNS", [])
            print("PREDICTOR_COLUMNS:", predictor_columns)
        else:
            print("¡El JSON cargado no es un diccionario!")
        self.COLUMNS_TRAINING = predictor_columns

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

    def force_predic(self):
        if not self.other_method or self.predictor_xgb_small is None or self.predictor_xgb_large is None:
            return None, [], []
        
        candidatos = {
            1: {
                'area': self.ref_area,
                'filled_volume': self.ref_filled_volume,
                'num_atm': self.ref_vecinos,
                'mean_distance': self.ref_mean_distance_single
            },
            2: {
                'area': self.ref_area_diva,
                'filled_volume': self.ref_filled_volume_diva,
                'num_atm': self.ref_vecinos_diva,
                'mean_distance': self.ref_mean_distance_diva
            },
            3: {
                'area': self.ref_area_tri,
                'filled_volume': self.ref_filled_volume_tri,
                'num_atm': self.ref_vecinos_tri,
                'mean_distance': self.ref_mean_distance_tri
            },
            4: {
                'area': self.ref_area_cua,
                'filled_volume': self.ref_filled_volume_cua,
                'num_atm': self.ref_vecinos_cua,
                'mean_distance': self.ref_mean_distance_cua
            },
            5: {
                'area': self.ref_area_cin,
                'num_atm': self.ref_vecinos_cin,
                'filled_volume': self.ref_filled_volume_cin,
                'mean_distance': self.ref_mean_distance_cin
            },
            self.predictor_rf_force:{
                'area':self.ref_area_cin_rf,
                'filled_volume': self.ref_filled_volume_cin_rf,
                'num_atm': self.ref_vecinos_cin_rf,
                'mean_distance':self.ref_mean_distance_cin_rf

            }
        }
        
        total_count = 0
        predictions = []
        errors = []
        parametro=0
        contador=0
        
        for i, (area, filled_volume, num_atm, mean_distance) in enumerate(
                zip(self.vector_area, self.vector_filled_volume, self.vector_num_atm, self.vector_mean_distance)):
            contador+=1
            errores_candidatos = {}
            for vac, refs in candidatos.items():
                error = 0
                if 'area' in refs:
                    error += abs(area - refs['area'])
                if 'filled_volume' in refs:
                    error += abs(filled_volume - refs['filled_volume'])
                if 'num_atm' in refs:
                    error += abs(num_atm - refs['num_atm'])
                if 'mean_distance' in refs:
                    error += abs(mean_distance - refs['mean_distance'])

                errores_candidatos[vac] = error-parametro*(contador)
            
            vacancias_pred = min(errores_candidatos, key=errores_candidatos.get)
            error_minimo = errores_candidatos[vacancias_pred]
            
            total_count += vacancias_pred
            predictions.append(vacancias_pred)
            if self.vector_true is not None:
                error_cluster = (vacancias_pred - self.vector_true[i]) ** 2
                errors.append(error_cluster)
            else:
                errors.append(None)
                
            print(f"[FORCE] Cluster {i}: Mejor vacancia predicha: {vacancias_pred} con error total: {error_minimo}. "
                f"Características: area={area}, filled_volume={filled_volume}, num_atm={num_atm}, mean_distance={mean_distance}")
        
        print(f"[FORCE] Total de vacancias predichas: {total_count}\n")
        return total_count, predictions, errors

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

    def run(self):
        self.load_data()
        total_xgb, pred_xgb, err_xgb = self.predict_xgb()
        self.force_predic()
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
        n = len(self.results["xgb"]["predictions"])
        df = pd.DataFrame({
            "cluster_id": list(range(n)),
           
            "xgb": self.results["xgb"]["predictions"]
          
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
        return self.results["xgb"]["predictions"]


    def export_predictions_accumulated(self, iteration, output_csv="outputs_tt/accumulated_predictions.csv"):
        """
        Exporta (o acumula) las predicciones de la iteración actual en un CSV general.
        Si el archivo ya existe, se añaden las filas sin sobrescribir el contenido previo.
        Se agrega una columna 'iteration' para identificar la corrida.
        """
        n = len(self.results["xgb"]["predictions"])
        df_iteration = pd.DataFrame({
            "iteration": [iteration] * n,
            "cluster_id": list(range(n)),
            "xgb": self.results["xgb"]["predictions"]
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
    
    error=np.array([0,0,0,0,0,0,0,0,0,0,0])
    for i in range(0,7):
        for j in range(0,10):
            with open("input_params.json", "r") as f:
                json_data = json.load(f)
                config = json_data["CONFIG"][0]
                defect_file_list = config['defect']
                relax = config['relax']
                radius_training = config['radius_training']
                radius = config['radius']
                with open('input_params.json', 'r') as file:
                    data = json.load(file)

                print(type(data))  # Asegurate que sea <class 'dict'>

                # Si es dict, accedé a la clave
                if isinstance(data, dict):
                    predictor_columns = data.get("PREDICTOR_COLUMNS", [])
                    print("PREDICTOR_COLUMNS:", predictor_columns)
                else:
                    print("¡El JSON cargado no es un diccionario!")

                smoothing_level_training = config['smoothing_level_training']
                 
                smoothing_level= config['smoothing level']
                other_method = config['other method']
                save_training = config['save_training']
                strees = config['strees']
            if config['activate_generate_relax']:
                from TrainingStepScript import CrystalStructureGenerator
                generator = CrystalStructureGenerator()
                generator.generate(repetitions=(10, 10, 10))
            if config['training_step']:
                from TrainingStepScript import TrainingProcessor
                processor = TrainingProcessor(relax, radius_training, radius, smoothing_level_training, strees, save_training)
                processor.run()

            ####################################################################################
            #como puedo exportar los resultados del error total junto con los valores de i y j para los que se dieron pero que no se sobreescriba sino que se acomulen en filas  una debajod e la otra
            contador = 0
            vacancias_predic = []
            error_predic = []
            reales = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
            resultados_totales = []

            for file_defect in defect_file_list:
                print(f"###################################{file_defect}##################################")
                processor = ClusterProcessor(file_defect)
                processor.run()
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
                    with open('input_params.json', 'r') as file:
                        data = json.load(file)

                    print(type(data))  # Asegurate que sea <class 'dict'>

                    # Si es dict, accedé a la clave
                    if isinstance(data, dict):
                        predictor_columns = data.get("PREDICTOR_COLUMNS", [])
                        print("PREDICTOR_COLUMNS:", predictor_columns)
                    else:
                        print("¡El JSON cargado no es un diccionario!")

                predictor_xgb_small = XGBoostVacancyPredictor("outputs.vfinder/training_small.json")
                predictor_xgb_large = XGBoostVacancyPredictor("outputs.vfinder/training_data.json")




                runner = VacancyPredictionRunner(
                    archivo=file_defect,
                    predictor_xgb_small=predictor_xgb_small,
                    predictor_xgb_large=predictor_xgb_large,
                    other_method=True
                )
                
                runner.run()
                runner.export_totals()
                vacancias = runner.export_predictions_per_cluster()
                vacancia_predicha = vacancias[0] if isinstance(vacancias, (list, np.ndarray)) else vacancias

                vacancia_floor = np.floor(vacancia_predicha)
                vacancia_ceil = np.ceil(vacancia_predicha)

                real_vacancy = reales[contador]

                error_floor = abs(vacancia_floor - real_vacancy)
                error_ceil = abs(vacancia_ceil - real_vacancy)

                if error_floor < error_ceil:
                    vacancia_ajustada = vacancia_floor
                    error = error_floor
                else:
                    vacancia_ajustada = vacancia_ceil
                    error = error_ceil

                vacancias_predic.append(vacancia_ajustada)
                error_predic.append(error)

                print(f"Predicción original: {vacancia_predicha} | Ajustada: {vacancia_ajustada} | Real: {real_vacancy} | Error: {error}")

                contador += 1

     

             
                if j>8:
                    
                    print(f"########################Error medio: {np.mean(error)/9}############################### [{i};(0 a {j})]")
                
                iteration = 1  
                runner.export_predictions_accumulated(iteration)
            contador=0
            with open("input_params.json", "r") as f:
                json_data = json.load(f)
            json_data["CONFIG"][0]['smoothing_level_training'] = j
            json_data["CONFIG"][0]['smoothing level'] = i
            print(f"Nuevo smoothing_level_training: {json_data['CONFIG'][0]['smoothing_level_training']}")
            with open("input_params.json", "w") as f:
                json.dump(json_data, f, indent=4)
            
            error_promedio = np.mean(error_predic)
            resultados_totales.append((i, j, error_promedio))

            print(f"@@@@@@@@@ Error promedio total: {error_promedio} @@@@@@@@@")
            
           

            csv_path = "outputs.vfinder/resultados_errores.csv"
            file_exists = os.path.isfile(csv_path)

            with open(csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if not file_exists:
                    writer.writerow(["i", "j", "error_promedio"])  # Solo escribimos encabezado si el archivo no existe
                writer.writerow([i, j, error_promedio])

            with open("outputs.vfinder/resultados_detallados.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["i", "j", "real", "prediccion", "error"])
                for idx in range(len(vacancias_predic)):
                    writer.writerow([i, j,  reales[idx], vacancias_predic[idx], error_predic[idx]])
            # Encontrar el mínimo error promedio
            mejor_i, mejor_j, mejor_error = min(resultados_totales, key=lambda x: x[2])

            # Agregar la fila al final del CSV
            with open("outputs.vfinder/resultados_errores.csv", mode='a', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([])
                writer.writerow(["MEJOR_COMBINACION", "i", "j", "error_promedio"])
                writer.writerow(["\n", mejor_i, mejor_j, mejor_error])
