import os
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (ExpressionSelectionModifier, DeleteSelectedModifier, 
                              ConstructSurfaceModifier, InvertSelectionModifier, 
                              AffineTransformationModifier)
from input_params import CONFIG
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
from input_params import PREDICTOR_COLUMNS
from sklearn.pipeline import Pipeline

from scipy.interpolate import interp1d


from scipy.interpolate import interp1d

class TrainingProcessor:
    def __init__(self, relax_file, radius_training, radius, smoothing_level_training, strees, save_training,output_dir="outputs.vfinder"):
        self.relax_file = relax_file
        self.radius_training = radius_training
        self.radius = radius
        self.smoothing_level_training = smoothing_level_training
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
        centro = TrainingProcessor.obtener_centro(self.relax_file)
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
        pipeline = import_file(self.ids_dump_file)
        data = pipeline.compute()
        particle_ids = data.particles["Particle Identifier"]
        return np.array(particle_ids).tolist()

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
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        return np.min(distances)
    
    def compute_mean_distance(self, data):
        positions = data.particles.positions
        center_of_mass = np.mean(positions, axis=0)
        distances = np.linalg.norm(positions - center_of_mass, axis=1)
        return np.mean(distances)
    def run_training(self):
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
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=condition_f))
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            pipeline_2.modifiers.append(ConstructSurfaceModifier(
                radius=self.radius,
                smoothing_level=self.smoothing_level_training,
                identify_regions=True,
                select_surface_particles=True
            ))
            data_2 = pipeline_2.compute()
            sm_elip = data_2.attributes.get('ConstructSurfaceMesh.surface_area', 0)
            filled_vol = data_2.attributes.get('ConstructSurfaceMesh.void_volume', 0)
            sm_mesh_training.append(sm_elip)
            vacancys.append(index + 1)
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            data_3 = pipeline_2.compute()
            #min_dist = self.compute_min_distance(data_3)
            mean_dist = self.compute_mean_distance(data_3)
            #min_distancias.append(min_dist)
            mean_distancias.append(mean_dist)
            vecinos.append(data_3.particles.count)
            filled_volumes.append(filled_vol)
            pipeline_2.modifiers.clear()
        
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


        

    def run(self):
        self.run_training()


def load_json_data(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)
    return pd.DataFrame(data)

class VacancyPredictorRF:
    def __init__(self, json_path="outputs.vfinder/training_data.json", predictor_columns=PREDICTOR_COLUMNS):
        self.json_path = json_path
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
                 predictor_columns=PREDICTOR_COLUMNS,
                 n_splits=5, random_state=42):
        self.training_data_path = training_data_path
        self.model_path = model_path
        self.n_splits = n_splits
        self.random_state = random_state
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
    def __init__(self, json_path="outputs.vfinder/training_data.json", predictor_columns=PREDICTOR_COLUMNS):
        self.json_path = json_path
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
    config = CONFIG[0]
    relax_file = config["relax"]
    save_training=config['save_training']
    radius_training = config["radius_training"]
    radius = config["radius"]
    smoothing_level_training = config["smoothing_level_training"]
    strees = [1.0, 1.0, 1.0]
  
    processor = TrainingProcessor(relax_file, radius_training, radius, smoothing_level_training, strees,save_training)
    processor.run()
