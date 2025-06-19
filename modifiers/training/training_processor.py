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
from training.utils import resolve_input_params_path
import math
import pandas as pd
from training.training_fingerstyle import StatisticsCalculator,DumpProcessor,FeatureExporter


from typing import Tuple

class TrainingProcessor:
    def __init__(
        self,
        radius_training: float = None,
        radius: float = None,
        smoothing_level_training: int = None,
        strees: tuple = (1.0, 1.0, 1.0),
        save_training: bool = True,
        relax_file: str = None,
        output_dir_json: str = "outputs/json",
        output_dir_csv: str = "outputs/csv",
        output_dir_dump: str = "outputs/dump",
        json_params_path: str = None
    ):
        """
        - relax_file: ruta al archivo LAMMPS dump relajado
        - radius_training: radio (float) para seleccionar Partículas
        - radius: radio (float) usado en ConstructSurfaceModifier
        - smoothing_level_training: smoothing level para ConstructSurfaceModifier en entrenamiento
        - strees: tupla de 3 floats para aplicar deformación afín (AffineTransformationModifier)
        - save_training: si True, extendemos training_data.json en output_dir
        - output_dir_*: carpetas de salida para dumps, jsons, csv
        - json_params_path: ruta explícita a input_params.json
        """
        # Rutas de salida
        self.output_dir_json = output_dir_json
        self.output_dir_csv = output_dir_csv
        self.output_dir_dump = output_dir_dump

        # Si no nos dieron ruta al JSON, la resolvemos dinámicamente
        if json_params_path is None:
            json_params_path = resolve_input_params_path(__file__, "input_params.json")

        # 1) Cargamos el JSON de parámetros
        with open(json_params_path, "r", encoding="utf-8") as f:
            all_params = json.load(f)
        if "CONFIG" not in all_params or not isinstance(all_params["CONFIG"], list) or len(all_params["CONFIG"]) == 0:
            raise KeyError("input_params.json debe contener la clave 'CONFIG' como lista no vacía.")
        config = all_params["CONFIG"][0]

        # 2) Extraemos de CONFIG los parámetros necesarios
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

        # 'strees' y 'save_training' opcionales
        self.strees = tuple(config.get("strees", strees))
        self.save_training = config.get("save_training", save_training)

        # 3) Rutas internas para outputs
        os.makedirs(self.output_dir_dump, exist_ok=True)
        os.makedirs(self.output_dir_json, exist_ok=True)
        os.makedirs(self.output_dir_csv, exist_ok=True)

        self.ids_dump_file = os.path.join(self.output_dir_dump, "ids.training.dump")
        self.training_results_file = os.path.join(self.output_dir_json, "training_data.json")


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
        cuya distancia al centro sea <= radius_training.
        """
        centro = TrainingProcessor.obtener_centro(self.relax_file)

        pipeline = import_file(self.relax_file)
        cond = (
            f"(Position.X - {centro[0]})*(Position.X - {centro[0]}) + "
            f"(Position.Y - {centro[1]})*(Position.Y - {centro[1]}) + "
            f"(Position.Z - {centro[2]})*(Position.Z - {centro[2]}) <= {self.radius_training**2}"
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


    def _read_ids_and_positions(self) -> Tuple[list, np.ndarray]:
        """
        Carga el dump 'ids.training.dump' y devuelve dos cosas:
          - lista de Particle Identifier (int)
          - array de posiciones Nx3 (x,y,z) alineadas en el mismo orden
        """
        pipeline = import_file(self.ids_dump_file)
        data = pipeline.compute()

        particle_ids = data.particles["Particle Identifier"][:].tolist()
        positions = data.particles.positions
        return particle_ids, positions


    @staticmethod
    def _order_ids_by_proximity(ids_list: list, positions: np.ndarray) -> list:
        """
        Recibe:
          - ids_list: [id1, id2, ..., idN]
          - positions: array Nx3 con las coordenadas correspondientes en el mismo orden

        Devuelve un nuevo listado de IDs ordenado de forma que cada ID
        sucesivo esté cerca espacialmente del anterior (algoritmo greedy:
        partimos del punto más cercano al centro, luego siempre elegimos
        el vecino más cercano que aún no esté en la lista ordenada).
        """
        N = len(ids_list)
        if N == 0:
            return []

        # 1) Calcular el centroide del conjunto de puntos
        centroid = np.mean(positions, axis=0)
        # 2) Encontrar qué punto (índice) está más cerca del centroide
        dists_to_centroid = np.linalg.norm(positions - centroid, axis=1)
        start_idx = int(np.argmin(dists_to_centroid))

        ordered_ids = [ids_list[start_idx]]
        ordered_positions = [positions[start_idx]]
        visited = set([start_idx])

        current_idx = start_idx
        for _ in range(N - 1):
            # Tomar las distancias del punto actual a todos los no visitados
            mask = np.array([i not in visited for i in range(N)])
            if not mask.any():
                break
            candidates_idx = np.nonzero(mask)[0]
            # Calcular distancias desde current_idx a cada candidato
            dists = np.linalg.norm(positions[candidates_idx] - positions[current_idx], axis=1)
            nearest_relative_idx = int(np.argmin(dists))
            next_idx = candidates_idx[nearest_relative_idx]

            ordered_ids.append(ids_list[next_idx])
            ordered_positions.append(positions[next_idx])
            visited.add(next_idx)
            current_idx = next_idx

        return ordered_ids


    @staticmethod
    def crear_condicion_ids(ids_eliminar: list) -> str:
        """
        Concatena: "ParticleIdentifier==id1 || ParticleIdentifier==id2 || ...".
        """
        return " || ".join([f"ParticleIdentifier=={pid}" for pid in ids_eliminar])


    def compute_mean_distance(self, data) -> float:
        posiciones = data.particles.positions
        centro_masa = np.mean(posiciones, axis=0)
        distancias = np.linalg.norm(posiciones - centro_masa, axis=1)
        return np.mean(distancias)


    def run_training(self):
        """
        1) Generar ids.training.dump con export_training_dump()
        2) Cargar IDs y posiciones de ese dump
        3) Ordenar esa lista de IDs por proximidad (cercanía espacial)
        4) Para k = 1..len(ids):
             - eliminar las primeras k IDs de la lista ordenada
             - computar área y volumen con ConstructSurfaceModifier
             - invertir selección, calcular distancia promedio y cluster_size
             - acumular resultados
        5) Guardar JSONs de training_data, training_small, key_single_vacancy y key_double_vacancy
        """

        # 1) Generar dump de IDs
        self.export_training_dump()

        # 2) Cargar lista de IDs y posiciones
        particle_ids_list, positions = self._read_ids_and_positions()

        # 3) Reordenar los IDs según proximidad
        ordered_ids = TrainingProcessor._order_ids_by_proximity(particle_ids_list, positions)

        # 4) Creamos un pipeline base desde el archivo relajado
        pipeline_2 = import_file(self.relax_file)
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
        mean_distancias = []
        i=0
        # 4.1) Recorremos cada “vacancia” de 1 hasta N, eliminando las k primeras partículas
        for idx in range(len(ordered_ids)):
            i += 1
            # Toma las IDs de 0..idx (inclusive) en la lista reordenada
            ids_a_eliminar = ordered_ids[: idx + 1]
            cond_f = TrainingProcessor.crear_condicion_ids(ids_a_eliminar)

            # 4.2) Eliminar las primeras idx+1 partículas
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=cond_f))
            pipeline_2.modifiers.append(DeleteSelectedModifier())

            # 4.3) Construir SurfaceMesh en lo que queda
            pipeline_2.modifiers.append(ConstructSurfaceModifier(
                radius=self.radius,
                smoothing_level=self.smoothing_level_training,
                identify_regions=True,
                select_surface_particles=True
            ))
            data_2 = pipeline_2.compute()
            try:
                export_file(
                    pipeline_2,
                    f"outputs/dump/{i}_training.dump",
                    "lammps/dump",
                    columns=[
                        "Particle Identifier",
                        "Particle Type",
                        "Position.X",
                        "Position.Y",
                        "Position.Z"
                    ]
                )
            except Exception as e:
                print("Error en export_training_dump:", e)
            # Área y volumen
            sm_elip  = data_2.attributes.get('ConstructSurfaceMesh.surface_area', 0)
            filled_v = data_2.attributes.get('ConstructSurfaceMesh.void_volume',  0)

            sm_mesh_training.append(sm_elip)
            filled_volumes.append(filled_v)
            vacancys.append(idx + 1)

            # 4.4) Invertir selección y calcular distancia promedio y cluster_size
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            try:
                export_file(
                    pipeline_2,
                    f"outputs/dump/vacancy_{i}_training.dump",
                    "lammps/dump",
                    columns=[
                        "Particle Identifier",
                        "Particle Type",
                        "Position.X",
                        "Position.Y",
                        "Position.Z"
                    ]
                )
            except Exception as e:
                print("Error en export_training_dump:", e)
            data_3 = pipeline_2.compute()

            mean_d = self.compute_mean_distance(data_3)
            mean_distancias.append(mean_d)
            vecinos.append(data_3.particles.count)

            # 4.5) Limpiar todos los modificadores para la próxima iteración
            pipeline_2.modifiers.clear()

        # 5) Preparar el diccionario a exportar
        datos_exportar = {
            "surface_area":    sm_mesh_training,
            "filled_volume":   filled_volumes,
            "vacancys":        vacancys,
            "cluster_size":    vecinos,
            "mean_distance":   mean_distancias
        }

        # 6) Verificar si existe training_data.json para extender
        default_keys = {
            "surface_area": [], "filled_volume": [],
            "vacancys": [], "cluster_size": [], "mean_distance": []
        }
        if os.path.exists(self.training_results_file):
            with open(self.training_results_file, "r", encoding="utf-8") as f:
                datos_previos = json.load(f)
            for key in default_keys:
                if key not in datos_previos:
                    datos_previos[key] = []
        else:
            datos_previos = default_keys

        if self.save_training:
            for key in datos_exportar:
                datos_previos[key].extend(datos_exportar[key])
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
        primeros_small = os.path.join(self.output_dir_json, "training_small.json")
        with open(primeros_small, "w", encoding="utf-8") as f:
            json.dump(primeros_datos, f, indent=4)

        # 8) Sobrescribir "training_data.json" con toda la data actual
        all_data_json = os.path.join(self.output_dir_json, "training_data.json")
        with open(all_data_json, "w", encoding="utf-8") as f:
            json.dump(datos_exportar, f, indent=4)

        # 9) "key_single_vacancy.json" con el primer punto
        primeros_datos_single = {
            "surface_area":  sm_mesh_training[:1],
            "filled_volume": filled_volumes[:1],
            "vacancys":      vacancys[:1],
            "cluster_size":  vecinos[:1],
            "mean_distance": mean_distancias[:1]
        }
        single_file = os.path.join(self.output_dir_json, "key_single_vacancy.json")
        with open(single_file, "w", encoding="utf-8") as f:
            json.dump(primeros_datos_single, f, indent=4)

        # 10) "key_double_vacancy.json" con el segundo punto
        primeros_datos_double = {
            "surface_area":  sm_mesh_training[1:2],
            "filled_volume": filled_volumes[1:2],
            "vacancys":      vacancys[1:2],
            "cluster_size":  vecinos[1:2],
            "mean_distance": mean_distancias[1:2]
        }
        double_file = os.path.join(self.output_dir_json, "key_double_vacancy.json")
        with open(double_file, "w", encoding="utf-8") as f:
            json.dump(primeros_datos_double, f, indent=4)


    def run(self):
        """
        Método público para invocar todo el proceso de entrenamiento.
        """
        self.run_training()
            # Lista de archivos .dump de entrenamiento (vacancias) a procesar
        dump_files = [
            "outputs/dump/vacancy_1_training.dump",
            "outputs/dump/vacancy_2_training.dump",
            "outputs/dump/vacancy_3_training.dump",
            "outputs/dump/vacancy_4_training.dump",
            "outputs/dump/vacancy_5_training.dump",
            "outputs/dump/vacancy_6_training.dump",
            "outputs/dump/vacancy_7_training.dump",
            "outputs/dump/vacancy_8_training.dump",
            "outputs/dump/vacancy_9_training.dump",
            "outputs/dump/vacancy_10_training.dump",
            "outputs/dump/vacancy_11_training.dump",
            "outputs/dump/vacancy_12_training.dump",
            "outputs/dump/vacancy_13_training.dump",
            "outputs/dump/vacancy_14_training.dump"
        ]
        output_csv_path = "outputs/csv/finger_data.csv"

        exporter = FeatureExporter(dump_files, output_csv_path)
        exporter.export()
