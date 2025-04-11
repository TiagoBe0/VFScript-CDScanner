import os
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (ExpressionSelectionModifier, DeleteSelectedModifier, 
                              ConstructSurfaceModifier, InvertSelectionModifier, 
                              AffineTransformationModifier)

import json
import math
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import xgboost as xgb
from sklearn.pipeline import Pipeline

from scipy.interpolate import interp1d


from scipy.interpolate import interp1d

class TrainingProcessor:
    def __init__(self, relax_file, radius_training, radius, smoothing_level_training, strees, save_training,output_dir="outputs.vfinder"):
        with open("input_params.json", "r") as f:
            json_data = json.load(f)
        config = json_data["CONFIG"][0]
        
        self.relax_file = config['relax']
        self.radius_training = config['radius_training']
        self.radius = config['radius']
        self.smoothing_level_training =config['smoothing_level_training']
        
   
        self.output_dir = output_dir
        self.save_training=save_training
        # Ensure output_dir is created and then create file paths
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)
        self.ids_dump_file = os.path.join(self.output_dir, "ids.training.dump")
        self.training_results_file = os.path.join(self.output_dir, "training_data.json")
        self.strees = strees

    @staticmethod
    def obtener_centro(file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
        box_bounds_index = None
        for i, line in enumerate(lines):
            if line.startswith("ITEM: BOX BOUNDS"):
                box_bounds_index = i
                break
        if box_bounds_index is None:
            raise ValueError("No se encontró la sección 'BOX BOUNDS' en el archivo.")
        x_bounds = lines[box_bounds_index + 1].split()
        y_bounds = lines[box_bounds_index + 2].split()
        z_bounds = lines[box_bounds_index + 3].split()
        x_min, x_max = map(float, x_bounds)
        y_min, y_max = map(float, y_bounds)
        z_min, z_max = map(float, z_bounds)
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        center_z = (z_min + z_max) / 2.0
        return center_x, center_y, center_z

    def export_training_dump(self):
        centro = self.obtener_centro(self.relax_file)
        print(f"Centro de la caja: {centro}")
        pipeline = import_file(self.relax_file)
        condition = (
            f"(Position.X - {centro[0]})*(Position.X - {centro[0]}) + "
            f"(Position.Y - {centro[1]})*(Position.Y - {centro[1]}) + "
            f"(Position.Z - {centro[2]})*(Position.Z - {centro[2]}) <= {self.radius_training * self.radius_training}"
        )
        pipeline.modifiers.append(ExpressionSelectionModifier(expression=condition))
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        try:
            export_file(pipeline, self.ids_dump_file, "lammps/dump", 
                        columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
            pipeline.modifiers.clear()
        except Exception as e:
            print("Error en export_training_dump:", e)

    def extract_particle_ids(self):
        try:
            pipeline = import_file(self.ids_dump_file)
            data = pipeline.compute()

            if not data.particles or len(data.particles) == 0:
                print("⚠️ No hay partículas en el archivo.")
                return []

            # Verificar atributos
            required_keys = {'Position', 'Particle Identifier'}
            available_keys = set(data.particles.keys())
            missing_keys = required_keys - available_keys
            
            if missing_keys:
                print(f"❌ Faltan atributos: {missing_keys}")
                return []

            positions = np.array(data.particles['Position'])
            particle_ids = np.array(data.particles['Particle Identifier'])
            print(f"ids desordenados: {particle_ids}")

            if len(positions) != len(particle_ids):
                print("⚠️ ¡Número de posiciones e IDs no coincide!")
                return []

            # Ordenar por distancia al PRIMER elemento (alternativa: centroide)
            reference_point = positions[0]
            distances = np.linalg.norm(positions - reference_point, axis=1)
            print(f"distancias:{distances}")
            sorted_ids = particle_ids[np.argsort(distances)]
            print(f"ids ordenados: {sorted_ids}")

            return sorted_ids.tolist()

        except Exception as e:
            print(f"❌ Error inesperado: {str(e)}")
            return []


    

    @staticmethod
    def crear_condicion_ids(ids_eliminar):
        return " || ".join([f"ParticleIdentifier=={id}" for id in ids_eliminar])

    def compute_max_distance(self, data):
        positions = data.particles.positions
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        return np.max(distances)

    def compute_min_distance(self, data):
        positions = data.particles.positions
        print(f"Posiciones: {positions}")
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        return np.min(distances)
    
    def compute_mean_distance(self, positions):
        
        center_of_mass = np.mean(positions, axis=0)
        print(f"Centro de masa: {center_of_mass}")
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        return np.mean(distances)
#necesito que se cree el directorio outputs.json en caso de que no exista antes de exportar el archivo.dump
    def run_training(self):
        os.makedirs("outputs.json", exist_ok=True)
        self.export_training_dump()
        particle_ids_list = self.extract_particle_ids()
        pipeline_2 = import_file(self.relax_file)
        pipeline_2.modifiers.append(AffineTransformationModifier(
            operate_on={'particles', 'cell'},
            transformation=[[self.strees[0], 0, 0, 0],
                            [0, self.strees[1], 0, 0],
                            [0, 0, self.strees[2], 0]]
        ))
        sm_mesh_training = []
        vacancys = []
        vecinos = []
        filled_volumes = []
        min_distancias = []
        mean_distancias = []
        for index in range(len(particle_ids_list)):
            ids_a_eliminar = particle_ids_list[:index + 1]
            condition_f = TrainingProcessor.crear_condicion_ids(ids_a_eliminar)
            
            # Aplicar modificadores
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=condition_f))
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            pipeline_2.modifiers.append(ConstructSurfaceModifier(
                radius=self.radius,
                smoothing_level=self.smoothing_level_training,
                identify_regions=True,
                select_surface_particles=True
            ))
            
            # Calcular datos de superficie
            data_2 = pipeline_2.compute()
            sm_elip = data_2.attributes.get('ConstructSurfaceMesh.surface_area', 0)
            filled_vol = data_2.attributes.get('ConstructSurfaceMesh.void_volume', 0)
            output_file = f"outputs.json/training_void_{index + 1}.dump"

            try:
                export_file(pipeline_2, output_file, "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z"])
                
            except Exception as e:
                print("error al exportar dump de entrenamiento.")
                pass            
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            data_3 = pipeline_2.compute()
            pipeline_2.modifiers.clear()
            positions = np.array(data_3.particles['Position']) if data_3.particles.count > 0 else np.empty((0,3))
            
            if len(positions) > 0:
                centro_de_masa = np.mean(positions, axis=0)
                distancias = np.linalg.norm(positions - centro_de_masa, axis=1)
                mean_dist = np.mean(distancias)
            else:
                mean_dist = 0.0  
            
            # Almacenar resultados
            sm_mesh_training.append(sm_elip)
            vacancys.append(index + 1)
            mean_distancias.append(mean_dist)
            vecinos.append(data_3.particles.count)
            filled_volumes.append(filled_vol)
            
            # Exportar (opcional)

        
        # Prepare data to export
        datos_exportar = {
            "surface_area": sm_mesh_training,
            "filled_volume": filled_volumes,
            "vacancys": vacancys,
            "cluster_size": vecinos,
            "mean_distance": mean_distancias
        }
        
        # Load and update previous training results only if self.save_training is True
        default_keys = {"surface_area": [], "filled_volume": [], "vacancys": [], "cluster_size": [], "mean_distance": []}
        if os.path.exists(self.training_results_file):
            with open(self.training_results_file, "r") as f:
                datos_previos = json.load(f)
            for key in default_keys:
                if key not in datos_previos:
                    datos_previos[key] = []
        else:
            datos_previos = default_keys
        
        if self.save_training:
            datos_previos["surface_area"].extend(sm_mesh_training)
            datos_previos["filled_volume"].extend(filled_volumes)
            datos_previos["vacancys"].extend(vacancys)
            datos_previos["cluster_size"].extend(vecinos)
            datos_previos["mean_distance"].extend(mean_distancias)
            with open(self.training_results_file, "w") as f:
                json.dump(datos_previos, f, indent=4)
        
        primeros_datos = {
            "surface_area": sm_mesh_training[:7],
            "filled_volume": filled_volumes[:7],
            "vacancys": vacancys[:7],
            "cluster_size": vecinos[:7],
            "mean_distance": mean_distancias[:7]
        }
        primeros_datos_file = os.path.join(os.path.dirname(self.training_results_file), "training_small.json")
        with open(primeros_datos_file, "w") as f:
            json.dump(primeros_datos, f, indent=4)
        
        primeros_datos = {
            "surface_area": sm_mesh_training,
            "filled_volume": filled_volumes,
            "vacancys": vacancys,
            "cluster_size": vecinos,
            "mean_distance": mean_distancias
        }
        primeros_datos_file = os.path.join(os.path.dirname(self.training_results_file), "training_data.json")
        with open(primeros_datos_file, "w") as f:
            json.dump(primeros_datos, f, indent=4)
                
        # Export separate file for one vacancy (first iteration)
        primeros_datos_single = {
            "surface_area": sm_mesh_training[:1],
            "filled_volume": filled_volumes[:1],
            "vacancys": vacancys[:1],
            "cluster_size": vecinos[:1],
            "mean_distance": mean_distancias[:1]
        }
        output_dir = os.path.dirname(self.training_results_file)
        single_file = os.path.join(output_dir, "key_single_vacancy.json")
        with open(single_file, "w") as f:
            json.dump(primeros_datos_single, f, indent=1)
        
        # Export separate file for two vacancies (second iteration)
        primeros_datos_double = {
            "surface_area": sm_mesh_training[1:2],
            "filled_volume": filled_volumes[1:2],
            "vacancys": vacancys[1:2],
            "cluster_size": vecinos[1:2],
            "mean_distance": mean_distancias[1:2]
        }
        double_file = os.path.join(output_dir, "key_double_vacancy.json")
        with open(double_file, "w") as f:
            json.dump(primeros_datos_double, f, indent=1)
        primeros_datos_cua = {
            "surface_area": sm_mesh_training[3:4],
            "filled_volume": filled_volumes[3:4],
            "vacancys": vacancys[3:4],
            "cluster_size": vecinos[3:4],
            "mean_distance": mean_distancias[3:4]
        }
        primeros_datos_cin = {
            "surface_area": sm_mesh_training[4:5],
            "filled_volume": filled_volumes[4:5],
            "vacancys": vacancys[4:5],
            "cluster_size": vecinos[4:5],
            "mean_distance": mean_distancias[4:5]
        }
        primeros_datos_six = {
            "surface_area": sm_mesh_training[5:6],
            "filled_volume": filled_volumes[5:6],
            "vacancys": vacancys[5:6],
            "cluster_size": vecinos[5:6],
            "mean_distance": mean_distancias[5:6]
        }
        primeros_datos_tri = {
            "surface_area": sm_mesh_training[2:3],
            "filled_volume": filled_volumes[2:3],
            "vacancys": vacancys[2:3],
            "cluster_size": vecinos[2:3],
            "mean_distance": mean_distancias[2:3]
        }
        double_file = os.path.join(output_dir, "key_double_vacancy.json")
        with open(double_file, "w") as f:
            json.dump(primeros_datos_double, f, indent=1)
        double_filee = os.path.join(output_dir, "key_tri_vacancy.json")
        with open(double_filee, "w") as f:
            json.dump(primeros_datos_tri, f, indent=1)
        double_fileee = os.path.join(output_dir, "key_cua_vacancy.json")
        with open(double_fileee, "w") as f:
            json.dump(primeros_datos_cua, f, indent=1)
        double_fileeee = os.path.join(output_dir, "key_cin_vacancy.json")
        with open(double_fileeee, "w") as f:
            json.dump(primeros_datos_cin, f, indent=1)
        double_fileeeee = os.path.join(output_dir, "key_six_vacancy.json")
        with open(double_fileeeee, "w") as f:
            json.dump(primeros_datos_six, f, indent=1)




        

    def run(self):
        self.run_training()


def load_json_data(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return pd.DataFrame(data)

class VacancyPredictorRF:
    def __init__(self, json_path="outputs.vfinder/training_data.json"):
        self.json_path = json_path
        with open('archivo.json', 'r') as file:
            data = json.load(file)

        print(type(data))  # Asegurate que sea <class 'dict'>

        # Si es dict, accedé a la clave
        if isinstance(data, dict):
            predictor_columns = data.get("PREDICTOR_COLUMNS", [])
            print("PREDICTOR_COLUMNS:", predictor_columns)
        else:
            print("¡El JSON cargado no es un diccionario!")
        self.columns = predictor_columns  # Se utilizan las columnas definidas en el input_param
        self.df = load_json_data(self.json_path)
        self.model = self._train_model()

    def _train_model(self):
        X = self.df[self.columns]
        y = self.df["vacancys"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return model

    def _round_up(self, x):
        return math.ceil(x)

    def predict_vacancies(self, **kwargs):
        data = {col: [kwargs[col]] for col in self.columns}
        nuevos_datos = pd.DataFrame(data)
        prediction = self.model.predict(nuevos_datos)[0]
        return self._round_up(prediction)
class XGBoostVacancyPredictor:
    def __init__(self, training_data_path="outputs.vfinder/training_data.json", 
                 model_path="outputs.json/xgboost_model.json", 
                 n_splits=5, random_state=42):
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.n_splits = n_splits
        self.random_state = random_state
        with open('input_params.json', 'r') as file:
            data = json.load(file)

        print(type(data))  # Asegurate que sea <class 'dict'>

        # Si es dict, accedé a la clave
        if isinstance(data, dict):
            predictor_columns = data.get("PREDICTOR_COLUMNS", [])
            print("PREDICTOR_COLUMNS:", predictor_columns)
        else:
            print("¡El JSON cargado no es un diccionario!")
        self.columns = predictor_columns  # Uso de las columnas del archivo de parámetros
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
        with open(self.training_data_path, "r") as f:
            data = json.load(f)
        feature_list = []
        for col in self.columns:
            if col in data:
                feature_list.append(data[col])
            else:
                raise ValueError(f"La columna '{col}' no se encuentra en los datos de entrenamiento.")
        X = np.column_stack(feature_list)
        y = np.array(data["vacancys"])
        X = self.scaler.fit_transform(X)
        n_samples = X.shape[0]
        n_splits = self.n_splits if n_samples >= self.n_splits else (n_samples if n_samples > 1 else 2)
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        scores = cross_val_score(self.model, X, y, scoring='neg_mean_squared_error', cv=kfold)
        mse_scores = -scores
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=self.random_state)
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        self.model.save_model(self.model_path)

    def predict(self, sample_input):
        sample_input = np.array(sample_input)
        sample_input = self.scaler.transform(sample_input)
        prediction = self.model.predict(sample_input)
        return prediction

import math
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

def load_json_data(json_path):
    import json
    with open(json_path, "r") as file:
        data = json.load(file)
    return pd.DataFrame(data)


class CrystalStructureGenerator:
    def __init__(self, config_file="input_params.json"):
        self.config_file = config_file
        self.structure_type = None
        self.lattice_parameter = None
        self.load_config()
    
    def load_config(self):
        """Carga la configuración desde el archivo JSON"""
        with open(self.config_file, "r") as f:
            json_data = json.load(f)
            config = json_data["CONFIG"][0]
            relax = config['generate_relax']
            self.structure_type = relax[0]
            self.lattice_parameter = float(relax[1])
    
    def generate_fcc(self, repetitions=(1,1,1)):
        """Genera estructura cristalina FCC"""
        # Posiciones atómicas base para FCC
        base_positions = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.5, 0.5, 1],
            [0.5, 1, 0.5],
            [1, 0.5, 0.5]
        ]) * self.lattice_parameter
        
        return self._replicate_structure(base_positions, repetitions)
    
    def generate_bcc(self, repetitions=(4,4,4)):
        """Genera estructura cristalina BCC"""
        # Posiciones atómicas base para BCC
        base_positions = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ]) * self.lattice_parameter
        
        return self._replicate_structure(base_positions, repetitions)
    
    def _replicate_structure(self, base_positions, repetitions):
        """Replica la estructura base según las repeticiones especificadas"""
        coordinates = []
        for i in range(repetitions[0]):
            for j in range(repetitions[1]):
                for k in range(repetitions[2]):
                    displacement = np.array([i, j, k]) * self.lattice_parameter
                    for atom in base_positions:
                        coordinates.append(atom + displacement)
        
        coordinates = np.array(coordinates)
        coordinates = np.unique(np.round(coordinates, decimals=6), axis=0)
        
        box_limits = (
            0.0, self.lattice_parameter * repetitions[0],
            0.0, self.lattice_parameter * repetitions[1],
            0.0, self.lattice_parameter * repetitions[2]
        )
        
        return coordinates, box_limits
    
    def save_to_dump(self, coordinates, box_limits, output_file):
        """Guarda la estructura en formato LAMMPS dump"""
        with open(output_file, "w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(coordinates)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{box_limits[0]} {box_limits[1]}\n")
            f.write(f"{box_limits[2]} {box_limits[3]}\n")
            f.write(f"{box_limits[4]} {box_limits[5]}\n")
            f.write("ITEM: ATOMS id type x y z\n")
            for i, coord in enumerate(coordinates, start=1):
                f.write(f"{i} 1 {coord[0]} {coord[1]} {coord[2]}\n")
    
    def generate(self, repetitions=(4,4,4)):
        """Genera la estructura según la configuración cargada"""
        if self.structure_type == "fcc":
            coordinates, box_limits = self.generate_fcc(repetitions)
            output_file = "inputs.dump/relax_structure.dump"
        elif self.structure_type == "bcc":
            coordinates, box_limits = self.generate_bcc(repetitions)
            output_file = "inputs.dump/relax_structure.dump"
        else:
            raise ValueError(f"Tipo de estructura no soportado: {self.structure_type}")
        
        self.save_to_dump(coordinates, box_limits, output_file)
        
        # Mostrar información de la estructura generada
        print(f"\nEstructura {self.structure_type.upper()} generada exitosamente")
        print(f"Archivo de salida: {output_file}")
        print(f"Parámetro de red: {self.lattice_parameter}")
        print(f"Celdas unitarias: {repetitions[0]}x{repetitions[1]}x{repetitions[2]}")
        print(f"Total de átomos: {len(coordinates)}")
        print(f"Límites de la caja:")
        print(f"  X: {box_limits[0]} - {box_limits[1]}")
        print(f"  Y: {box_limits[2]} - {box_limits[3]}")
        print(f"  Z: {box_limits[4]} - {box_limits[5]}")




class VacancyPredictor:
    def __init__(self, json_path="outputs.vfinder/training_data.json"):
        self.json_path = json_path
        # Se usa siempre la columna "surface_area" para este modelo
        self.columns = ["surface_area"]
        self.df = load_json_data(self.json_path)
        self.model = self._train_model()

    def _train_model(self):
        X = self.df[self.columns]
        y = self.df["vacancys"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        return model

    def _round_positive(self, x):
        return math.ceil(x) if x > 0 else math.ceil(-x)

    def predict_vacancies(self, **kwargs):
        nuevos_datos = pd.DataFrame({col: [kwargs[col]] for col in self.columns})
        prediction = self.model.predict(nuevos_datos)[0]
        return self._round_positive(prediction)





from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
import math


class VacancyPredictorMLP:
    def __init__(self, json_path="outputs.vfinder/training_data.json"):
        self.json_path = json_path
        with open('input_params.json.json', 'r') as file:
            data = json.load(file)

        print(type(data))  # Asegurate que sea <class 'dict'>

        # Si es dict, accedé a la clave
        if isinstance(data, dict):
            predictor_columns = data.get("PREDICTOR_COLUMNS", [])
            print("PREDICTOR_COLUMNS:", predictor_columns)
        else:
            print("¡El JSON cargado no es un diccionario!")
        self.columns = predictor_columns  # Uso de las columnas definidas en los parámetros
        self.df = load_json_data(self.json_path)
        self.model = self._train_model()

    def _train_model(self):
        X = self.df[self.columns]
        y = self.df["vacancys"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        # Creamos un pipeline que primero escala los datos y luego entrena el modelo MLP
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(128, 64),
                activation='relu',
                solver='adam',
                learning_rate_init=0.01,
                max_iter=1000,
                early_stopping=True,
                n_iter_no_change=20,
                random_state=42
            ))
        ])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        print("MSE del modelo MLP:", mse)
        return pipeline

    def _round_up(self, x):
        return math.ceil(x) if x > 0 else math.ceil(-x)

    def predict_vacancies(self, **kwargs):
        # Se crea un DataFrame con los datos de entrada; el pipeline se encarga de escalarlos
        data = pd.DataFrame({col: [kwargs[col]] for col in self.columns})
        prediction = self.model.predict(data)[0]
        return self._round_up(prediction)


class SyntheticDataGenerator:
    def __init__(self, data, num_points=100, interpolation_kind='linear'):

        self.data = data
        self.num_points = num_points
        self.interpolation_kind = interpolation_kind
        
        required_keys = ["surface_area", "filled_volume", "vacancys", "cluster_size","mean_distance"]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"La clave '{key}' no se encuentra en los datos.")
        
        self.vacancias = np.array(self.data["vacancys"])
    
    def generate(self):
        vac_new = np.linspace(self.vacancias.min(), self.vacancias.max(), self.num_points)
        
        interp_sm = interp1d(self.vacancias, self.data["surface_area"], kind=self.interpolation_kind)
        sm_new = interp_sm(vac_new)
        interp_mdistance = interp1d(self.vacancias, self.data["mean_distance"], kind=self.interpolation_kind)
        mdistance_new = interp_sm(vac_new)
        
        interp_filled = interp1d(self.vacancias, self.data["filled_volume"], kind=self.interpolation_kind)
        filled_new = interp_filled(vac_new)
        
        interp_vecinos = interp1d(self.vacancias, self.data[ "cluster_size"], kind=self.interpolation_kind)
        vecinos_new = np.round(interp_vecinos(vac_new)).astype(int)
        
        synthetic_data = {
            "surface_area": sm_new.tolist(),
            "filled_volume": filled_new.tolist(),
            "vacancys": vac_new.tolist(),
            "cluster_size": vecinos_new.tolist(),

            "mean_distance": mdistance_new.tolist()
        }
        return synthetic_data

    def export_to_json(self, output_path, data):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Datos exportados a {output_path}")

from scipy.optimize import brentq

class VacancyPredictorCurve:
    def __init__(self, training_json_path, csv_path, degree=3):
        """
        Inicializa el predictor con los paths de los datos de entrenamiento y del CSV,
        además del grado del polinomio a ajustar.
        """
        self.training_json_path = training_json_path
        self.csv_path = csv_path
        self.degree = degree
        self.training_data = None
        self.vacancias_train = None
        self.surface_area_train = None
        self.poly = None
        self.min_area_train = None
        self.max_area_train = None

    def load_training_data(self, as_dataframe=False):
        """
        Carga los datos de entrenamiento desde un archivo JSON.
        Si as_dataframe es True, retorna un DataFrame de pandas.
        """
        with open(self.training_json_path, "r") as f:
            data = json.load(f)
        if as_dataframe:
            data = pd.DataFrame(data)
        self.training_data = data
        return self.training_data

    def prepare_training_data(self):
        """
        Prepara los datos de entrenamiento extrayendo las columnas 'vacancias' y 'sm_mesh_training'
        a partir del tercer elemento, y define el rango de áreas de entrenamiento.
        """
        if self.training_data is None:
            raise ValueError("Los datos de entrenamiento no han sido cargados.")
        self.vacancias_train = self.training_data["vacancys"].iloc[2:]
        self.surface_area_train = self.training_data["surface_area"].iloc[2:]
        self.min_area_train = self.surface_area_train.min()
        self.max_area_train = self.surface_area_train.max()

    def fit_curve(self):
        """
        Ajusta un polinomio de grado 'degree' a los datos de entrenamiento.
        """
        if self.vacancias_train is None or self.surface_area_train is None:
            raise ValueError("Los datos de entrenamiento no han sido preparados.")
        coef = np.polyfit(self.vacancias_train, self.surface_area_train, deg=self.degree)
        self.poly = np.poly1d(coef)
        return self.poly

    def predict_vacancies_from_area(self, observed_area, vacancy_range=(1, 9), area_range=(None, None)):
        """
        Predice el número de vacancias para un área observada.
        Si el área está fuera del rango de entrenamiento se 'clampa' al valor mínimo o máximo.
        """
        if self.poly is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta fit_curve primero.")
        min_area, max_area = area_range
        if min_area is not None and observed_area < min_area:
            return vacancy_range[0]
        if max_area is not None and observed_area > max_area:
            return vacancy_range[1]
        def f(x):
            return self.poly(x) - observed_area
        try:
            vac_pred = brentq(f, vacancy_range[0], vacancy_range[1])
            return vac_pred
        except ValueError:
            return None

    def plot_training_fit(self):
        """
        Genera un gráfico de los datos de entrenamiento y el polinomio ajustado.
        """
        if self.vacancias_train is None or self.surface_area_train is None:
            raise ValueError("Los datos de entrenamiento no han sido preparados.")
        if self.poly is None:
            raise ValueError("El modelo no ha sido entrenado. Ejecuta fit_curve primero.")
        x_fit = np.linspace(self.vacancias_train.min(), self.vacancias_train.max(), 100)
        y_fit = self.poly(x_fit)


    def predict_from_csv(self):
        """
        Carga el CSV con los datos de defecto y predice las vacancias para cada valor de 'area'.
        Retorna el DataFrame actualizado con las predicciones.
        """
        csv_data = pd.read_csv(self.csv_path)
        predictions = []
        for idx, row in csv_data.iterrows():
            observed_area = row["area"]
            pred = self.predict_vacancies_from_area(
                observed_area,
                vacancy_range=(1, 9),
                area_range=(self.min_area_train, self.max_area_train)
            )
            predictions.append(pred)
        csv_data["predicted_vacancies"] = predictions
        return csv_data




class SyntheticDataGenerator:
    def __init__(self, data, num_points=100, interpolation_kind='linear'):

        self.data = data
        self.num_points = num_points
        self.interpolation_kind = interpolation_kind
        
        required_keys = ["surface_area", "filled_volume", "vacancys", "cluster_size","mean_distance"]
        for key in required_keys:
            if key not in self.data:
                raise ValueError(f"La clave '{key}' no se encuentra en los datos.")
        
        self.vacancias = np.array(self.data["vacancys"])
    
    def generate(self):
        vac_new = np.linspace(self.vacancias.min(), self.vacancias.max(), self.num_points)
        
        interp_sm = interp1d(self.vacancias, self.data["surface_area"], kind=self.interpolation_kind)
        sm_new = interp_sm(vac_new)
        
        interp_filled = interp1d(self.vacancias, self.data["filled_volume"], kind=self.interpolation_kind)
        filled_new = interp_filled(vac_new)
        
        interp_vecinos = interp1d(self.vacancias, self.data["cluster_size"], kind=self.interpolation_kind)
        vecinos_new = np.round(interp_vecinos(vac_new)).astype(int)
        interp_mean = interp1d(self.vacancias, self.data["mean_distance"], kind=self.interpolation_kind)
        mean_new = np.round(interp_mean(vac_new)).astype(int)
        
        synthetic_data = {
            "surface_area": sm_new.tolist(),
            "filled_volume": filled_new.tolist(),
            "vacancys": vac_new.tolist(),
            "cluster_size": vecinos_new.tolist(),

            "mean_distance": mean_new.tolist()
        }
        return synthetic_data

    def export_to_json(self, output_path, data):
        with open(output_path, "w") as f:
            json.dump(data, f, indent=4)
        print(f"Datos exportados a {output_path}")
if __name__ == "__main__":
    with open("input_params.json", "r") as f:
        json_data = json.load(f)
    config = json_data["CONFIG"][0]    
    relax_file = config["relax"]
    save_training=config['save_training']
    radius_training = config["radius_training"]
    radius = config["radius"]
    smoothing_level_training = config["smoothing_level_training"]
    strees = [1.0, 1.0, 1.0]
  
    processor = TrainingProcessor(relax_file, radius_training, radius, smoothing_level_training, strees,save_training)
    processor.run()
