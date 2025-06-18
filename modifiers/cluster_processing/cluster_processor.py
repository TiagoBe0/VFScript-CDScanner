import os
import json
import math
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import (
    DeleteSelectedModifier,
    InvertSelectionModifier,
    ExpressionSelectionModifier,
    ClusterAnalysisModifier,
    ConstructSurfaceModifier
)
from utils.utilidades_clustering import UtilidadesClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples

def merge_clusters(labels, c1, c2):
    new_labels = np.copy(labels)
    new_labels[new_labels == c2] = c1
    return new_labels

def compute_dispersion(coords, labels):
    dispersion_dict = {}
    for c in np.unique(labels):
        mask = (labels == c)
        if not np.any(mask):
            dispersion_dict[c] = np.nan
            continue
        cluster_coords = coords[mask]
        center_of_mass = cluster_coords.mean(axis=0)
        distances = np.linalg.norm(cluster_coords - center_of_mass, axis=1)
        dispersion_dict[c] = distances.std()
    return dispersion_dict

def silhouette_mean(coords, labels):
    sil_vals = silhouette_samples(coords, labels)
    return np.mean(sil_vals)

def try_all_merges(coords, labels):
    clusters_unique = np.unique(labels)
    results = []
    for i in range(len(clusters_unique)):
        for j in range(i + 1, len(clusters_unique)):
            c1 = clusters_unique[i]
            c2 = clusters_unique[j]
            fused_labels = merge_clusters(labels, c1, c2)
            new_unique = np.unique(fused_labels)
            if len(new_unique) == 1:
                continue
            s_mean = silhouette_mean(coords, fused_labels)
            disp_dict = compute_dispersion(coords, fused_labels)
            disp_sum = np.nansum(list(disp_dict.values()))
            results.append(((c1, c2), fused_labels, s_mean, disp_dict, disp_sum))
    return results

def get_worst_cluster(dispersion_dict):
    worst_cluster = None
    max_disp = -1
    for c_label, d_val in dispersion_dict.items():
        if d_val > max_disp:
            max_disp = d_val
            worst_cluster = c_label
    return worst_cluster, max_disp

def kmeans_three_points(coords):
    center_of_mass = coords.mean(axis=0)
    distances = np.linalg.norm(coords - center_of_mass, axis=1)
    far_idxs = np.argsort(distances)[-2:]
    initial_centers = np.vstack([
        center_of_mass,
        coords[far_idxs[0]],
        coords[far_idxs[1]]
    ])
    kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, random_state=42)
    kmeans.fit(coords)
    sub_labels = kmeans.labels_
    sub_sil = np.mean(silhouette_samples(coords, sub_labels))
    sub_disp_dict = compute_dispersion(coords, sub_labels)
    sub_disp_sum = np.nansum(list(sub_disp_dict.values()))
    return sub_labels, sub_sil, sub_disp_dict, sub_disp_sum

def iterative_fusion_and_subdivision(coords, init_labels, threshold=1.2, max_iterations=10):
    labels = np.copy(init_labels)
    iteration = 0
    while iteration < max_iterations:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            break
        merge_candidates = try_all_merges(coords, labels)
        if not merge_candidates:
            break
        best_merge = min(merge_candidates, key=lambda x: x[4])
        (_, fused_labels, _, fused_disp_dict, _) = best_merge
        labels = fused_labels
        worst_cluster, max_disp = get_worst_cluster(fused_disp_dict)
        if max_disp > threshold:
            mask = (labels == worst_cluster)
            coords_worst = coords[mask]
            sub_labels, _, _, _ = kmeans_three_points(coords_worst)
            offset = worst_cluster * 10
            new_sub_labels = offset + sub_labels
            new_labels_global = np.copy(labels)
            new_labels_global[mask] = new_sub_labels
            labels = new_labels_global
        iteration += 1
    return labels

class ClusterProcessor:
    def __init__(self, defect: str, json_params_path: str = None):
        """
        Si json_params_path no se pasa explícitamente, busca
        'input_params.json' un nivel arriba de este archivo.
        """

        # 1) Si no me pasan ruta, construyo la que apunte a '../input_params.json'
        if json_params_path is None:
            # __file__ = '/home/santi/Desktop/VFS-DOC/modifiers/cluster_processing/cluster_processor.py'
            carpeta_actual = os.path.dirname(__file__)
            # carpeta_raiz = '/home/santi/Desktop/VFS-DOC/modifiers'
            carpeta_raiz = os.path.dirname(carpeta_actual)
            # Ahora apunto a: '/home/santi/Desktop/VFS-DOC/modifiers/input_params.json'
            json_params_path = os.path.join(carpeta_raiz, "input_params.json")

        # 2) Intento abrir el JSON en la ruta calculada
        try:
            with open(json_params_path, "r", encoding="utf-8") as f:
                all_params = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No encontré el archivo de parámetros en: {json_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"El JSON en {json_params_path} está mal formado: {e}")

        # 3) Continuar extrayendo CONFIG[0], etc.
        if "CONFIG" not in all_params or not isinstance(all_params["CONFIG"], list) or len(all_params["CONFIG"]) == 0:
            raise KeyError("input_params.json debe contener una lista bajo la clave 'CONFIG'.")
        self.config = all_params["CONFIG"][0]

        # 4) Extraer los parámetros que necesites (radius, smoothing_level, cutoff, …)
        #    Ejemplo:
        rad = self.config.get("radius")
        if isinstance(rad, list):
            self.radio_sonda = rad
        else:
            self.radio_sonda = [rad]

        self.smoothing_leveled = self.config["smoothing_level"]
        self.cutoff_radius = self.config["cutoff"]
        self.nombre_archivo = self.config["defect"]
        # … resto de tu lógica …
        # 3) Crear carpetas de salida
        self.outputs_dump = "outputs/dump"
        self.outputs_json = "outputs/json"
        os.makedirs(self.outputs_dump, exist_ok=True)
        os.makedirs(self.outputs_json, exist_ok=True)

    def run(self):
        """
        1) Aplica ConstructSurfaceModifier a todo el dump inicial para identificar
           todas las partículas y crear un dump intermedio ("key_areas.dump").
        2) Escribe un JSON con el número de clusters detectados.
        3) Para cada cluster, genera un dump separado bajo outputs.dump/key_area_i.dump.
        """
        # 1) Pipeline para superficie inicial
        pipeline = import_file(self.nombre_archivo)
        # Si quieres iterar varios radios, podrías elegir el primero o todos; aquí tomo el primero:
        r = self.radio_sonda[0]
        pipeline.modifiers.append(
            ConstructSurfaceModifier(
                radius=r,
                smoothing_level=self.smoothing_leveled,
                identify_regions=True,
                select_surface_particles=True
            )
        )
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(
            ClusterAnalysisModifier(
                cutoff=self.cutoff_radius,
                sort_by_size=True,
                unwrap_particles=True,
                compute_com=True
            )
        )
        data = pipeline.compute()

        # 2) Número total de clusters detectados
        num_clusters = data.attributes["ClusterAnalysis.cluster_count"]
        datos_clusters = {"num_clusters": num_clusters}
        clusters_json_path = os.path.join(self.outputs_json, "clusters.json")
        with open(clusters_json_path, "w", encoding="utf-8") as archivo:
            json.dump(datos_clusters, archivo, indent=4)

        # 3) Exportar el dump con todas las partículas y su etiqueta de cluster
        key_areas_dump_path = os.path.join(self.outputs_dump, "key_areas.dump")
        try:
            export_file(
                pipeline,
                key_areas_dump_path,
                "lammps/dump",
                columns=[
                    "Particle Identifier",
                    "Particle Type",
                    "Position.X",
                    "Position.Y",
                    "Position.Z",
                    "Cluster"
                ]
            )
            pipeline.modifiers.clear()
        except Exception:
            # Ignoramos errores en la exportación
            pass

        # 4) Para cada uno de los clusters, filtramos y creamos un dump independiente
        for i in range(1, num_clusters + 1):
            cluster_expr = f"Cluster=={i}"
            pipeline_2 = import_file(key_areas_dump_path)
            pipeline_2.modifiers.append(
                ClusterAnalysisModifier(
                    cutoff=self.cutoff_radius,
                    cluster_coloring=True,
                    unwrap_particles=True,
                    sort_by_size=True
                )
            )
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=cluster_expr))
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            output_file = os.path.join(self.outputs_dump, f"key_area_{i}.dump")
            try:
                export_file(
                    pipeline_2,
                    output_file,
                    "lammps/dump",
                    columns=[
                        "Particle Identifier",
                        "Particle Type",
                        "Position.X",
                        "Position.Y",
                        "Position.Z",
                        "Cluster"
                    ]
                )
                pipeline_2.modifiers.clear()
            except Exception:
                # Si falla para algún cluster, continuamos con el siguiente
                pass

        print(f"Número de áreas clave encontradas: {num_clusters}")

    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        """
        Extrae y devuelve todas las líneas del archivo hasta 'ITEM: ATOMS' (incluyéndola).
        """
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception:
            pass
        return encabezado


class ClusterProcessorMachine:
    def __init__(
        self,
        file_path: str,
        json_params_path: str = "input_params.json"
    ):
        """
        Carga parámetros desde input_params.json si es necesario (p.ej. min_atoms),
        pero los valores por defecto aquí se quedan igual que antes.
        Se espera que input_params.json contenga:
        {
          "CONFIG": [
            {
              "divisions_of_cluster": <valor>,
              ...otros valores...
            }
          ]
        }
        """
        self.file_path = file_path

        # 1) Cargar input_params.json para 'divisions_of_cluster' o 'cluster tolerance'
        try:
            with open(json_params_path, "r", encoding="utf-8") as f:
                all_params = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de parámetros: {json_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear {json_params_path}: {e}")

        if "CONFIG" not in all_params or not isinstance(all_params["CONFIG"], list) or len(all_params["CONFIG"]) == 0:
            raise KeyError("El JSON no contiene una lista válida en la clave 'CONFIG'.")
        config = all_params["CONFIG"][0]

        # Umbral para reagrupar/subdividir clusters (threshold)
        # En tu JSON se llama "cluster tolerance"
        self.threshold = config.get("cluster tolerance", 1.2)
        # Número máximo de iteraciones para fusionar/subdividir
        self.max_iterations = config.get("iteraciones_clusterig", 10)

        # min_atoms: si no viene en el JSON, se resuelve desde el JSON de key_single_vacancy
        self.min_atoms = UtilidadesClustering.cargar_min_atoms("outputs/json/key_single_vacancy.json")

        # 2) Cargar datos del dump: matriz total y encabezado
        self.matriz_total = UtilidadesClustering.extraer_datos_completos(file_path)
        self.header = UtilidadesClustering.extraer_encabezado(file_path)

        # 3) También extraer posiciones y etiquetas iniciales vía OVITO
        pipeline = import_file(file_path)
        data = pipeline.compute()
        self.coords = data.particles.positions
        self.init_labels = data.particles["Cluster"].array

    def process_clusters(self):
        """
        Realiza la fusión/subdivisión iterativa de clusters a partir de las etiquetas iniciales.
        Sobrescribe la columna 5 de self.matriz_total con las etiquetas finales remapeadas a [0..n-1].
        """
        self.final_labels = iterative_fusion_and_subdivision(
            self.coords,
            self.init_labels,
            threshold=self.threshold,
            max_iterations=self.max_iterations
        )

        # Remapear etiquetas a un rango continuo 0..(n_clusters-1)
        unique = np.unique(self.final_labels)
        mapping = {old: new for new, old in enumerate(unique)}
        self.final_labels = np.vectorize(mapping.get)(self.final_labels)

        if self.matriz_total.shape[0] == self.final_labels.shape[0]:
            self.matriz_total[:, 5] = self.final_labels
        else:
            print("¡Atención! La cantidad de filas en la matriz y las etiquetas no coincide.")

    def export_updated_file(self, output_file: str = None):
        """
        Exporta self.matriz_total (con columna de cluster actualizada) a disco.
        Si no se especifica nombre, sobrescribe el mismo file_path.
        """
        if output_file is None:
            output_file = self.file_path

        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
        except Exception as e:
            print(f"Error al exportar {output_file}: {e}")

            
class ClusterDumpProcessor:
    """
    Carga un dump “crítico” y aplica KMeans inicial (3 clusters).
    Luego sobreescribe la columna “Cluster” en la matriz total.
    """
    def __init__(self, file_path: str, decimals: int = 5):
        self.file_path = file_path
        self.matriz_total = None
        self.header = None
        self.subset = None
        self.divisions_of_cluster = 3

    def load_data(self):
        self.matriz_total = UtilidadesClustering.extraer_datos_completos(self.file_path)
        self.header = UtilidadesClustering.extraer_encabezado(self.file_path)
        if self.matriz_total.size == 0:
            raise ValueError(f"No se pudieron extraer datos de {self.file_path}")
        self.subset = self.matriz_total[:, 2:5]

    def calcular_dispersion(self, points: np.ndarray) -> float:
        if points.shape[0] == 0:
            return 0.0
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        return np.mean(distances)

    def process_clusters(self):
        self.load_data()
        centro_masa_global = np.mean(self.subset, axis=0)
        p1, p2, distancia_maxima = self.find_farthest_points(self.subset)
        labels = self.aplicar_kmeans(self.subset, p1, p2, centro_masa_global, n_clusters=3)
        if labels.shape[0] != self.matriz_total.shape[0]:
            raise ValueError("Número de etiquetas != filas de la matriz total.")
        self.matriz_total[:, 5] = labels

    def ejecutar_silhotte(self):
        lista_criticos = UtilidadesClustering.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
        for arch in lista_criticos:
            processor = ClusterProcessorMachine(arch, threshold=1.2, max_iterations=10)
            processor.process_clusters()
            processor.export_updated_file()

    def separar_coordenadas_por_cluster(self) -> dict:
        if self.matriz_total is None:
            raise ValueError("Datos no cargados. Ejecutar load_data() primero.")
        clusters_dict = {}
        etiquetas_unicas = np.unique(self.matriz_total[:, 5])
        for etiqueta in etiquetas_unicas:
            coords = self.matriz_total[self.matriz_total[:, 5] == etiqueta][:, 2:5]
            clusters_dict[int(etiqueta)] = coords
        return clusters_dict

    def export_updated_file(self, output_file: str = None):
        if output_file is None:
            output_file = f"{self.file_path}_actualizado.txt"
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
        except Exception:
            pass

    @staticmethod
    def extraer_datos_completos(file_path: str) -> np.ndarray:
        return UtilidadesClustering.extraer_datos_completos(file_path)

    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        return UtilidadesClustering.extraer_encabezado(file_path)

    @staticmethod
    def cargar_lista_archivos_criticos(json_path: str) -> list:
        return UtilidadesClustering.cargar_lista_archivos_criticos(json_path)

    @staticmethod
    def aplicar_kmeans(
        coordenadas: np.ndarray,
        p1,
        p2,
        centro_masa_global,
        n_clusters: int
    ) -> np.ndarray:
        if n_clusters == 2:
            init_centers = np.array([p1, p2])
        elif n_clusters == 3:
            init_centers = np.array([p1, p2, centro_masa_global])
        else:
            raise ValueError("Solo se admite n_clusters = 2 o 3.")
        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_centers,
            n_init=1,
            max_iter=300,
            tol=1,
            random_state=42
        )
        etiquetas = kmeans.fit_predict(coordenadas)
        return etiquetas

    @staticmethod
    def find_farthest_points(coordenadas: np.ndarray) -> tuple:
        pts = np.array(coordenadas)
        n = pts.shape[0]
        if n < 2:
            return None, None, 0
        diffs = pts[:, None, :] - pts[None, :, :]
        distancias = np.sqrt((diffs ** 2).sum(axis=-1))
        idx = np.unravel_index(np.argmax(distancias), distancias.shape)
        distancia_maxima = distancias[idx]
        punto1 = pts[idx[0]]
        punto2 = pts[idx[1]]
        return punto1, punto2, distancia_maxima