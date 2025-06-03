# modifiers/training/training_processor.py

import os
import json
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (
    ExpressionSelectionModifier,
    DeleteSelectedModifier,
    ConstructSurfaceModifier,
    InvertSelectionModifier,
    AffineTransformationModifier
)
from training.utils import resolve_input_params_path  # función de utils.py
import math
import pandas as pd


class TrainingProcessor:
    def __init__(
        self,
        radius_training: float = None,
        radius: float = None,
        smoothing_level_training: int = None,
        strees: tuple = (1.0, 1.0, 1.0),
        save_training: bool = True,
        relax_file: str = None,
        output_dir: str = "outputs.vfinder",
        json_params_path: str = None
    ):
        """
        Parámetros obtenidos desde input_params.json (si no se pasan explícitamente,
        buscamos input_params.json en el nivel superior a este módulo).
        
        - relax_file: ruta al archivo LAMMPS dump relajado
        - radius_training: radio (float) para seleccionar partículas de entrenamiento
        - radius: radio (float) usado en ConstructSurfaceModifier
        - smoothing_level_training: smoothing level para ConstructSurfaceModifier en entrenamiento
        - strees: tupla de 3 floats para aplicar deformación afín (AffineTransformationModifier)
        - save_training: si True, extendemos training_data.json en output_dir
        - output_dir: carpeta donde crear ids.training.dump y training_data.json
        - json_params_path: ruta explícita a input_params.json (si None, se calcula automáticamente)
        """

        # Si no nos dieron ruta al JSON, la resolvemos dinámicamente
        if json_params_path is None:
            json_params_path = resolve_input_params_path(__file__, "input_params.json")

        # 1) Cargamos el JSON de parámetros
        with open(json_params_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)
        if "CONFIG" not in all_params or not isinstance(all_params["CONFIG"], list) or len(all_params["CONFIG"]) == 0:
            raise KeyError("input_params.json debe contener la clave 'CONFIG' como lista no vacía.")
        config = all_params["CONFIG"][0]

        # 2) Extraemos del JSON los parámetros que vienen en CONFIG:
        #    El script original tomaba:
        #       relax_file, radius_training, radius, smoothing_level_training, strees, save_training
        #    de CONFIG. Aquí validamos su existencia.
        try:
            self.relax_file = config["relax"]
        except KeyError:
            raise KeyError("Falta la clave 'relax' en CONFIG de input_params.json")

        try:
            self.radius_training = config["radius_training"]
        except KeyError:
            raise KeyError("Falta la clave 'radius_training' en CONFIG de input_params.json")

        try:
            self.radius = config["radius"]
        except KeyError:
            raise KeyError("Falta la clave 'radius' en CONFIG de input_params.json")

        try:
            self.smoothing_level_training = config["smoothing_level_training"]
        except KeyError:
            raise KeyError("Falta la clave 'smoothing_level_training' en CONFIG de input_params.json")

        # `strees` y `save_training` también podrían venir de CONFIG
        self.strees = tuple(config.get("strees", strees))
        self.save_training = config.get("save_training", save_training)

        # 3) Configuramos output_dir (si no existe, lo creamos)
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir, exist_ok=True)

        # 4) Definimos rutas internas para dumps y JSON de resultados
        self.ids_dump_file = os.path.join(self.output_dir, "ids.training.dump")
        self.training_results_file = os.path.join(self.output_dir, "training_data.json")


    @staticmethod
    def obtener_centro(file_path: str) -> tuple:
        """
        Lee un dump LAMMPS y calcula el centro geométrico en base a BOX BOUNDS.
        Retorna (center_x, center_y, center_z).
        """
        with open(file_path, 'r', encoding="utf-8") as f:
            lines = f.readlines()

        box_bounds_index = None
        for i, line in enumerate(lines):
            if line.startswith("ITEM: BOX BOUNDS"):
                box_bounds_index = i
                break
        if box_bounds_index is None:
            raise ValueError("No se encontró la sección 'BOX BOUNDS' en el archivo de input.")

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
        """
        Genera un dump llamado 'ids.training.dump' con todas las partículas
        cuya distancia al centro sea <= radius_training. (Las partículas cercanas
        se eliminan, y luego se exportan las restantes).
        """
        centro = TrainingProcessor.obtener_centro(self.relax_file)

        pipeline = import_file(self.relax_file)
        # Condición: (x - cx)^2 + (y - cy)^2 + (z - cz)^2 <= radius_training^2
        cond = (
            f"(Position.X - {centro[0]})*(Position.X - {centro[0]}) + "
            f"(Position.Y - {centro[1]})*(Position.Y - {centro[1]}) + "
            f"(Position.Z - {centro[2]})*(Position.Z - {centro[2]}) <= {self.radius_training ** 2}"
        )
        pipeline.modifiers.append(ExpressionSelectionModifier(expression=cond))
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        try:
            export_file(
                pipeline,
                self.ids_dump_file,
                "lammps/dump",
                columns=[
                    "Particle Identifier",
                    "Particle Type",
                    "Position.X",
                    "Position.Y",
                    "Position.Z"
                ]
            )
            pipeline.modifiers.clear()
        except Exception as e:
            print("Error en export_training_dump:", e)


    def extract_particle_ids(self) -> list:
        """
        Carga el dump 'ids.training.dump' generado y retorna la lista de IDs de las partículas.
        """
        pipeline = import_file(self.ids_dump_file)
        data = pipeline.compute()
        particle_ids = data.particles["Particle Identifier"]
        return particle_ids[:].tolist()


    @staticmethod
    def crear_condicion_ids(ids_eliminar: list) -> str:
        """
        A partir de una lista de IDs de partículas, construye una expresión LAMMPS
        tipo "ParticleIdentifier==id1 || ParticleIdentifier==id2 || ...".
        """
        return " || ".join([f"ParticleIdentifier=={pid}" for pid in ids_eliminar])


    def compute_max_distance(self, data) -> float:
        posiciones = data.particles.positions
        centro_masa = np.mean(posiciones, axis=0)
        distancias = np.linalg.norm(posiciones - centro_masa, axis=1)
        return np.max(distancias)


    def compute_min_distance(self, data) -> float:
        posiciones = data.particles.positions
        centro_masa = np.mean(posiciones, axis=0)
        distancias = np.linalg.norm(posiciones - centro_masa, axis=1)
        return np.min(distancias)


    def compute_mean_distance(self, data) -> float:
        posiciones = data.particles.positions
        centro_masa = np.mean(posiciones, axis=0)
        distancias = np.linalg.norm(posiciones - centro_masa, axis=1)
        return np.mean(distancias)


    def run_training(self):
        """
        Ciclo principal de entrenamiento:
        1) export_training_dump()  → genera ids.training.dump
        2) extrae lista de IDs
        3) Para cada k en [1..len(ids)]:
             - elimina las k primeras partículas
             - construye SurfaceMesh (ConstructSurfaceModifier)
             - obtiene area y filled_volume
             - luego inverte selección, computa distancia promedio y cuenta vecinos
             - guarda esos valores en listas
        4) Construye JSON con toda la data:
            {
              "surface_area": [...],
              "filled_volume": [...],
              "vacancys": [...],
              "cluster_size": [...],
              "mean_distance": [...]
            }
           → lo guarda en `training_data.json` (extendiendo si ya existe).
        5) También escribe ‘training_small.json’ (primeros 7 puntos),
           'key_single_vacancy.json' (primer punto) y
           'key_double_vacancy.json' (segundo punto).
        """

        # 1) Generar dump de IDs
        self.export_training_dump()

        # 2) Extraer lista de IDs
        particle_ids_list = self.extract_particle_ids()

        # 3) Crear un pipeline base a partir del archivo relajado
        pipeline_2 = import_file(self.relax_file)
        # Aplicar deformación afín si es necesario
        pipeline_2.modifiers.append(AffineTransformationModifier(
            operate_on={'particles', 'cell'},
            transformation=[
                [self.strees[0], 0, 0, 0],
                [0, self.strees[1], 0, 0],
                [0, 0, self.strees[2], 0]
            ]
        ))

        # Listas auxiliares que llenaremos
        sm_mesh_training = []
        vacancys        = []
        vecinos         = []
        filled_volumes  = []
        min_distancias  = []
        mean_distancias = []

        # 4) Ciclo sobre cada número de vacancias (1, 2, 3, …)
        for idx in range(len(particle_ids_list)):
            ids_a_eliminar = particle_ids_list[: idx + 1]
            cond_f = TrainingProcessor.crear_condicion_ids(ids_a_eliminar)

            # 4.1) Eliminar las primeras idx+1 partículas
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=cond_f))
            pipeline_2.modifiers.append(DeleteSelectedModifier())

            # 4.2) Construir SurfaceMesh en lo que queda
            pipeline_2.modifiers.append(ConstructSurfaceModifier(
                radius=self.radius,
                smoothing_level=self.smoothing_level_training,
                identify_regions=True,
                select_surface_particles=True
            ))
            data_2 = pipeline_2.compute()

            # Área y volumen
            sm_elip  = data_2.attributes.get('ConstructSurfaceMesh.surface_area', 0)
            filled_v = data_2.attributes.get('ConstructSurfaceMesh.void_volume',  0)

            sm_mesh_training.append(sm_elip)
            filled_volumes.append(filled_v)
            vacancys.append(idx + 1)

            # 4.3) Ahora invertimos la selección para computar distancias y vecinos
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            data_3 = pipeline_2.compute()

            # Diferentes distancias (min, mean, max). Puedes descomentar si las necesitas
            # min_d = self.compute_min_distance(data_3)
            mean_d = self.compute_mean_distance(data_3)
            # max_d = self.compute_max_distance(data_3)

            # Guardamos únicamente mean_distance y cluster_size (número de partículas)
            mean_distancias.append(mean_d)
            vecinos.append(data_3.particles.count)

            # 4.4) Limpiamos todos los modificadores para volver al “estado limpio”
            pipeline_2.modifiers.clear()

        # 5) Preparamos el diccionario a exportar
        datos_exportar = {
            "surface_area":    sm_mesh_training,
            "filled_volume":   filled_volumes,
            "vacancys":        vacancys,
            "cluster_size":    vecinos,
            "mean_distance":   mean_distancias
        }

        # 6) Si ya existe un training_data.json y save_training=True, extendemos
        default_keys = { "surface_area": [], "filled_volume": [], 
                         "vacancys": [], "cluster_size": [], "mean_distance": [] }

        if os.path.exists(self.training_results_file):
            with open(self.training_results_file, "r", encoding="utf-8") as f:
                datos_previos = json.load(f)
            for key in default_keys:
                if key not in datos_previos:
                    datos_previos[key] = []
        else:
            datos_previos = default_keys

        if self.save_training:
            datos_previos["surface_area"].extend(   sm_mesh_training)
            datos_previos["filled_volume"].extend(   filled_volumes)
            datos_previos["vacancys"].extend(        vacancys)
            datos_previos["cluster_size"].extend(    vecinos)
            datos_previos["mean_distance"].extend(   mean_distancias)
            with open(self.training_results_file, "w", encoding="utf-8") as f:
                json.dump(datos_previos, f, indent=4)

        # 7) Escribir "training_small.json" con los primeros 7 puntos
        primeros_datos = {
            "surface_area":    sm_mesh_training[:7],
            "filled_volume":   filled_volumes[:7],
            "vacancys":        vacancys[:7],
            "cluster_size":    vecinos[:7],
            "mean_distance":   mean_distancias[:7]
        }
        primeros_small = os.path.join(
            os.path.dirname(self.training_results_file),
            "training_small.json"
        )
        with open(primeros_small, "w", encoding="utf-8") as f:
            json.dump(primeros_datos, f, indent=4)

        # 8) Escribir "training_data.json" (sobrescribe con toda la data)
        all_data_json = os.path.join(
            os.path.dirname(self.training_results_file),
            "training_data.json"
        )
        with open(all_data_json, "w", encoding="utf-8") as f:
            json.dump(datos_exportar, f, indent=4)

        # 9) "key_single_vacancy.json" con solo el primer punto
        primeros_datos_single = {
            "surface_area":  sm_mesh_training[:1],
            "filled_volume": filled_volumes[:1],
            "vacancys":      vacancys[:1],
            "cluster_size":  vecinos[:1],
            "mean_distance": mean_distancias[:1]
        }
        single_file = os.path.join(os.path.dirname(self.training_results_file), "key_single_vacancy.json")
        with open(single_file, "w", encoding="utf-8") as f:
            json.dump(primeros_datos_single, f, indent=4)

        # 10) "key_double_vacancy.json" con solo el segundo punto
        primeros_datos_double = {
            "surface_area":  sm_mesh_training[1:2],
            "filled_volume": filled_volumes[1:2],
            "vacancys":      vacancys[1:2],
            "cluster_size":  vecinos[1:2],
            "mean_distance": mean_distancias[1:2]
        }
        double_file = os.path.join(os.path.dirname(self.training_results_file), "key_double_vacancy.json")
        with open(double_file, "w", encoding="utf-8") as f:
            json.dump(primeros_datos_double, f, indent=4)


    def run(self):
        """
        Método público para invocar todo el proceso de entrenamiento.
        """
        self.run_training()
