import os
import json
import numpy as np
from ovito.io import import_file
from ovito.modifiers import ConstructSurfaceModifier
from utils.utilidades_clustering import UtilidadesClustering

class SurfaceProcessor:
    def __init__(
        self,
        json_params_path: str = "modifiers/input_params.json",
        key_archivos_path: str = "outputs.json/key_archivos.json",
        threshold_file: str = "outputs.vfinder/key_single_vacancy.json"
    ):
        """
        Ahora los parámetros de entrada (radius, smoothing_level, etc.) se leen
        desde `input_params.json`. Se espera que ese JSON tenga al menos estas claves:
           {
             "CONFIG": [
               {
                 "radius": <número o lista de números>,
                 "smoothing_level": <entero>,
                 ...
               }
             ],
             "PREDICTOR_COLUMNS": [...]
           }
        """
        # 1) Cargar todo el JSON de entrada
        try:
            with open(json_params_path, "r", encoding="utf-8") as f:
                all_params = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de parámetros: {json_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON en {json_params_path}: {e}")

        # 2) Extraer la configuración principal (suponemos que está en el primer elemento de CONFIG)
        if "CONFIG" not in all_params or not isinstance(all_params["CONFIG"], list) or len(all_params["CONFIG"]) == 0:
            raise KeyError("El JSON no contiene una lista válida bajo la clave 'CONFIG'.")
        self.config = all_params["CONFIG"][0]

        # 3) Del JSON de configuración, tomamos los valores que antes venían de CONFIG[0]
        #    Cambiamos "smoothing level" por "smoothing_level", según tu JSON
        try:
            self.smoothing_level = self.config["smoothing_level"]
        except KeyError:
            raise KeyError("Falta la clave 'smoothing_level' en el JSON de CONFIG.")

        # 4) La lista de radios a iterar (antes estaba hardcodeada). Si en el JSON viene solo un número,
        #    lo convertimos a lista; si viene lista, lo usamos directamente. En tu input_params.json
        #    aparece "radius": 2, así que por defecto creamos [2]. Si quisieras probar múltiples radios,
        #    podrías cambiar en el JSON a algo como "radius": [2, 3, 4].
        radius = self.config["radius"]
         # Si no viene definido, mantenemos la lista por defecto
        self.radi = [2, 3, 4, 5, 6, 7, 8, 9, 10]

        # 5) Rutas a otros JSONs clave
        self.key_archivos_path = key_archivos_path
        try:
            with open(self.key_archivos_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de clusters: {self.key_archivos_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON en {self.key_archivos_path}: {e}")

        # 6) Extraemos lista de dumps finales
        self.clusters_final = self.data.get("clusters_final", [])
        if not isinstance(self.clusters_final, list):
            raise ValueError(f"'clusters_final' debe ser una lista en {self.key_archivos_path}")

        # 7) Cargar umbrales mínimos de área y volumen desde el JSON de threshold
        try:
            with open(threshold_file, "r", encoding="utf-8") as f:
                threshold_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de umbrales: {threshold_file}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear JSON en {threshold_file}: {e}")

        # El JSON de umbrales tiene claves "surface_area" y "filled_volume" como listas
        try:
            self.min_area_threshold = threshold_data["surface_area"][0] / 2
        except (KeyError, IndexError):
            raise KeyError("Falta 'surface_area' o no es una lista con al menos un elemento en el JSON de umbrales.")

        try:
            self.min_filled_volume_threshold = threshold_data["filled_volume"][0] / 2
        except (KeyError, IndexError):
            raise KeyError("Falta 'filled_volume' o no es una lista con al menos un elemento en el JSON de umbrales.")

        # 8) Inicializamos la matriz de resultados vacía
        self.results_matrix = None


    def process_surface_for_file(self, archivo):
        """
        Para cada dump (archivo) intenta todos los radios en self.radi,
        calcula área, volumen y distancia promedio, y devuelve el pipeline
        que maximiza el área. Si no supera los umbrales, devuelve Nones.
        """
        best_area = 0
        best_filled_volume = 0
        best_radius = None
        best_pipeline = None
        best_avg_distance = 0

        for r in self.radi:
            pipeline = import_file(archivo)
            pipeline.modifiers.append(
                ConstructSurfaceModifier(
                    radius=r,
                    smoothing_level=self.smoothing_level,
                    identify_regions=True,
                    select_surface_particles=True
                )
            )
            data = pipeline.compute()
            cluster_size = data.particles.count

            # 1) Área de la malla (si existe)
            try:
                area = data.attributes['ConstructSurfaceMesh.surface_area']
            except Exception:
                area = 0

            # 2) Volumen relleno
            try:
                filled_volume = data.attributes['ConstructSurfaceMesh.filled_volume']
            except Exception:
                filled_volume = 0

            # 3) Distancia promedio al centro de masa
            positions = data.particles.positions
            if positions.shape[0] > 0:
                center = np.mean(positions, axis=0)
                avg_distance = np.mean(np.linalg.norm(positions - center, axis=1))
            else:
                avg_distance = 0

            # 4) Actualizar mejor candidato
            if area > best_area:
                best_area = area
                best_filled_volume = filled_volume
                best_radius = r
                best_pipeline = pipeline
                best_avg_distance = avg_distance

        # 5) Revisamos umbrales mínimos
        if best_area < self.min_area_threshold or best_filled_volume < self.min_filled_volume_threshold:
            return None, None, None, None, None, None

        return best_pipeline, best_radius, best_area, best_filled_volume, cluster_size, best_avg_distance


    def process_all_files(self):
        """
        Itera sobre self.clusters_final y aplica process_surface_for_file() a cada dump.
        Devuelve una matriz NumPy de los resultados válidos.
        """
        results = []
        for archivo in self.clusters_final:
            bp, br, ba, fv, num_atm, avg_dist = self.process_surface_for_file(archivo)
            if bp is not None:
                results.append([archivo, br, ba, fv, num_atm, avg_dist])

        self.results_matrix = np.array(results, dtype=object)
        return self.results_matrix


    def export_results(self, output_csv: str = "outputs.json/resultados_procesados.csv"):
        """
        Guarda self.results_matrix en un CSV. Si aún no se calculó, invoca a process_all_files().
        """
        if self.results_matrix is None:
            self.process_all_files()

        # Asegurarse de que exista el directorio de destino
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        np.savetxt(
            output_csv,
            self.results_matrix,
            delimiter=",",
            fmt="%s",
            header="archivo,mejor_radio,area,filled_volume,num_atm,mean_distance",
            comments=""
        )
