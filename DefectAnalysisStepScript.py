import os
import json
import numpy as np
from ovito.io import import_file, export_file
from ovito.modifiers import ExpressionSelectionModifier, DeleteSelectedModifier, ClusterAnalysisModifier, ConstructSurfaceModifier, InvertSelectionModifier
from input_params import CONFIG,PREDICTOR_COLUMNS
from sklearn.metrics import silhouette_samples
from sklearn.cluster import KMeans
import math
class SurfaceProcessor:
    def __init__(self, config=CONFIG[0], json_path="outputs.json/key_archivos.json", radi=None, threshold_file="outputs.vfinder/key_single_vacancy.json"):
        self.config = config
        self.smoothing_level = config["smoothing level"]
        self.radi = radi if radi is not None else [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.json_path = json_path
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.clusters_final = self.data.get("clusters_final", [])
        self.results_matrix = None
        with open(threshold_file, "r", encoding="utf-8") as f:
            threshold_data = json.load(f)
        self.min_area_threshold = threshold_data["surface_area"][0] / 2
        self.min_filled_volume_threshold = threshold_data["filled_volume"][0] / 2

    def process_surface_for_file(self, archivo):
        best_area = 0
        best_filled_volume = 0
        best_radius = None
        best_pipeline = None
        best_avg_distance = 0
        for r in self.radi:
            pipeline = import_file(archivo)
            pipeline.modifiers.append(ConstructSurfaceModifier(
                radius=r,
                smoothing_level=self.smoothing_level,
                identify_regions=True,
                select_surface_particles=True
            ))
            data = pipeline.compute()
            cluster_size = data.particles.count
            try:
                area = data.attributes['ConstructSurfaceMesh.surface_area']
            except Exception as e:
                area = 0
            try:
                filled_volume = data.attributes['ConstructSurfaceMesh.filled_volume']
            except Exception as e:
                filled_volume = 0
            # Calcular la distancia promedio al centro de masa del clúster
            positions = data.particles.positions
            if positions.shape[0] > 0:
                center = np.mean(positions, axis=0)
                avg_distance = np.mean(np.linalg.norm(positions - center, axis=1))
            else:
                avg_distance = 0
            if area > best_area:
                best_area = area
                best_filled_volume = filled_volume
                best_radius = r
                best_pipeline = pipeline
                best_avg_distance = avg_distance
        if best_area < self.min_area_threshold or best_filled_volume < self.min_filled_volume_threshold:
            return None, None, None, None, None, None
        return best_pipeline, best_radius, best_area, best_filled_volume, cluster_size, best_avg_distance

    def process_all_files(self):
        results = []
        for archivo in self.clusters_final:
            bp, br, ba, fv, num_atm, avg_dist = self.process_surface_for_file(archivo)
            if bp is not None:
                results.append([archivo, br, ba, fv, num_atm, avg_dist])
        self.results_matrix = np.array(results)
        return self.results_matrix

    def export_results(self, output_csv="outputs.json/resultados_procesados.csv"):
        if self.results_matrix is None:
            self.process_all_files()
        np.savetxt(output_csv, self.results_matrix, delimiter=",", fmt="%s",
                   header="archivo,mejor_radio,area,filled_volume,num_atm,mean_distance", comments="")


class ClusterDumpProcessor:
    def __init__(self, file_path: str, decimals: int = 5):
        self.file_path = file_path
        self.matriz_total = None
        self.header = None
        self.subset = None
        self.divisions_of_cluster = CONFIG[0]['divisions_of_cluster']

    def load_data(self):
        self.matriz_total = self.extraer_datos_completos(self.file_path)
        self.header = self.extraer_encabezado(self.file_path)
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
        centro_masa_global = np.mean(self.subset, axis=0)
        p1, p2, distancia_maxima = self.find_farthest_points(self.subset)
        dispersion = self.calcular_dispersion(self.subset)
        threshold = self.divisions_of_cluster
        etiquetas = self.aplicar_kmeans(self.subset, p1, p2, centro_masa_global, n_clusters=3)
        if etiquetas.shape[0] != self.matriz_total.shape[0]:
            raise ValueError("El número de etiquetas no coincide con la matriz total.")
        self.matriz_total[:, 5] = etiquetas

    def ejecutar_silhotte(self):
        lista_criticos = UtilidadesClustering.cargar_lista_archivos_criticos("outputs.json/key_archivos.json")
        processor = ClusterProcessor(self.archivo, decimals=4, threshold=1.2, max_iterations=10)
        processor.process_clusters()
        processor.export_updated_file()

    def separar_coordenadas_por_cluster(self) -> dict:
        if self.matriz_total is None:
            raise ValueError("Los datos no han sido cargados. Ejecuta load_data() primero.")
        clusters_dict = {0, 1, 2}
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
        except Exception as e:
            pass

    @staticmethod
    def cargar_lista_archivos_criticos(json_path: str) -> list:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            return datos.get("clusters_criticos", [])
        except FileNotFoundError:
            return []
        except json.JSONDecodeError as e:
            return []

    @staticmethod
    def extraer_datos_completos(file_path: str) -> np.ndarray:
        datos = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return np.array([])
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break
        if start_index is None:
            return np.array([])
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                id_val = int(parts[0])
                type_val = int(parts[1])
                x = round(float(parts[2]), 5)
                y = round(float(parts[3]), 5)
                z = round(float(parts[4]), 5)
                cluster_val = int(parts[5])
                datos.append([id_val, type_val, x, y, z, cluster_val])
            except ValueError:
                continue
        return np.array(datos)

    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception as e:
            pass
        return encabezado

    @staticmethod
    def aplicar_kmeans(coordenadas: np.ndarray, p1, p2, centro_masa_global, n_clusters: int) -> np.ndarray:
        from sklearn.cluster import KMeans
        if n_clusters == 2:
            init_centers = np.array([p1, p2])
        elif n_clusters == 3:
            init_centers = np.array([p1, p2, centro_masa_global])
        else:
            raise ValueError("Solo se admite n_clusters igual a 2 o 3.")
        kmeans = KMeans(n_clusters=n_clusters,
                        init=init_centers,
                        n_init=1,
                        max_iter=300,
                        tol=1,
                        random_state=42)
        etiquetas = kmeans.fit_predict(coordenadas)
        return etiquetas

    @staticmethod
    def find_farthest_points(coordenadas: np.ndarray):
        pts = np.array(coordenadas)
        n = pts.shape[0]
        if n < 2:
            return None, None, 0
        diffs = pts[:, None, :] - pts[None, :, :]
        distancias = np.sqrt(np.sum(diffs**2, axis=-1))
        idx = np.unravel_index(np.argmax(distancias), distancias.shape)
        distancia_maxima = distancias[idx]
        punto1 = pts[idx[0]]
        punto2 = pts[idx[1]]
        return punto1, punto2, distancia_maxima

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
    initial_centers = np.vstack([center_of_mass, coords[far_idxs[0]], coords[far_idxs[1]]])
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
        (pair, fused_labels, fused_sil, fused_disp_dict, fused_disp_sum) = best_merge
        labels = fused_labels
        worst_cluster, max_disp = get_worst_cluster(fused_disp_dict)
        if max_disp > threshold:
            mask = (labels == worst_cluster)
            coords_worst = coords[mask]
            sub_labels, sub_sil, sub_disp_dict, sub_disp_sum = kmeans_three_points(coords_worst)
            offset = worst_cluster * 10
            new_sub_labels = offset + sub_labels
            new_labels_global = np.copy(labels)
            new_labels_global[mask] = new_sub_labels
            labels = new_labels_global
        iteration += 1
    return labels

class ClusterProcessor:
    def __init__(self,defect):
        self.configuracion = CONFIG[0]
        self.nombre_archivo = defect
        self.radio_sonda = self.configuracion['radius']
        self.smoothing_leveled = self.configuracion['smoothing level']
        self.cutoff_radius = self.configuracion['cutoff radius']
        self.outputs_dump = "outputs.dump"
        self.outputs_json = "outputs.json"
        os.makedirs(self.outputs_dump, exist_ok=True)
        os.makedirs(self.outputs_json, exist_ok=True)
    
    def run(self):
        pipeline = import_file(self.nombre_archivo)
        pipeline.modifiers.append(ConstructSurfaceModifier(radius=self.radio_sonda, smoothing_level=self.smoothing_leveled, identify_regions=True, select_surface_particles=True))
        pipeline.modifiers.append(InvertSelectionModifier())
        pipeline.modifiers.append(DeleteSelectedModifier())
        pipeline.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, sort_by_size=True, unwrap_particles=True, compute_com=True))
        data = pipeline.compute()
        num_clusters = data.attributes["ClusterAnalysis.cluster_count"]
        datos_clusters = {"num_clusters": num_clusters}
        clusters_json_path = os.path.join(self.outputs_json, "clusters.json")
        with open(clusters_json_path, "w") as archivo:
            json.dump(datos_clusters, archivo, indent=4)
        key_areas_dump_path = os.path.join(self.outputs_dump, "key_areas.dump")
        try:
            export_file(pipeline, key_areas_dump_path, "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
            pipeline.modifiers.clear()
        except Exception as e:
            pass
        clusters = [f"Cluster=={i}" for i in range(1, num_clusters + 1)]
        for i, cluster_expr in enumerate(clusters, start=1):
            pipeline_2 = import_file(key_areas_dump_path)
            pipeline_2.modifiers.append(ClusterAnalysisModifier(cutoff=self.cutoff_radius, cluster_coloring=True, unwrap_particles=True, sort_by_size=True))
            pipeline_2.modifiers.append(ExpressionSelectionModifier(expression=cluster_expr))
            pipeline_2.modifiers.append(InvertSelectionModifier())
            pipeline_2.modifiers.append(DeleteSelectedModifier())
            output_file = os.path.join(self.outputs_dump, f"key_area_{i}.dump")
            try:
                export_file(pipeline_2, output_file, "lammps/dump", columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                pipeline_2.modifiers.clear()
            except Exception as e:
                pass
        print(f"Número de áreas clave encontradas: {num_clusters}")
    
    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception as e:
            pass
        return encabezado
    


class UtilidadesClustering:
    @staticmethod
    def cargar_lista_archivos_criticos(json_path: str) -> list:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            return datos.get("clusters_criticos", [])
        except FileNotFoundError:
            print(f"El archivo {json_path} no existe.")
            return []
        except json.JSONDecodeError as e:
            print(f"Error al decodificar el archivo JSON: {e}")
            return []
    
    @staticmethod
    def extraer_datos_completos(file_path: str, decimals: int = 5) -> np.ndarray:
        datos = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            print(f"No se encontró el archivo: {file_path}")
            return np.array([])
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break
        if start_index is None:
            print(f"No se encontró la sección 'ITEM: ATOMS' en {file_path}.")
            return np.array([])
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                id_val = int(parts[0])
                type_val = int(parts[1])
                x = round(float(parts[2]), decimals)
                y = round(float(parts[3]), decimals)
                z = round(float(parts[4]), decimals)
                cluster_val = int(parts[5])
                datos.append([id_val, type_val, x, y, z, cluster_val])
            except ValueError:
                continue
        return np.array(datos)
    
    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception as e:
            print(f"Error al extraer encabezado de {file_path}: {e}")
        return encabezado
    
    @staticmethod
    def cargar_min_atoms(json_path: str) -> int:
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            vecinos = datos.get("cluster_size", [])
            if vecinos and isinstance(vecinos, list):
                return int(vecinos[0])
            else:
                print("No se encontró el valor de 'vecinos' en el archivo JSON, se usa valor por defecto 14.")
                return 14
        except Exception as e:
            print(f"Error al cargar {json_path}: {e}. Se usa valor por defecto 14.")
            return 14


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
        for j in range(i+1, len(clusters_unique)):
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
    initial_centers = np.vstack([center_of_mass, coords[far_idxs[0]], coords[far_idxs[1]]])
    kmeans = KMeans(n_clusters=3, init=initial_centers, n_init=1, random_state=42)
    kmeans.fit(coords)
    sub_labels = kmeans.labels_
    sub_sil = np.mean(silhouette_samples(coords, sub_labels))
    sub_disp_dict = compute_dispersion(coords, sub_labels)
    sub_disp_sum = np.nansum(list(sub_disp_dict.values()))
    return sub_labels, sub_sil, sub_disp_dict, sub_disp_sum

def kmeans_default(coords):
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(coords)
    sub_labels = kmeans.labels_
    sub_sil = np.mean(silhouette_samples(coords, sub_labels))
    sub_disp_dict = compute_dispersion(coords, sub_labels)
    sub_disp_sum = np.nansum(list(sub_disp_dict.values()))
    return sub_labels, sub_sil, sub_disp_dict, sub_disp_sum
def iterative_fusion_and_subdivision(coords, init_labels, threshold=1.5, max_iterations=10, min_atoms=14):
    labels = np.copy(init_labels)
    iteration = 0
    while iteration < max_iterations:
        unique_labels = np.unique(labels)
        if len(unique_labels) <= 1:
            # Sólo queda 1 clúster.
            break
        merge_candidates = try_all_merges(coords, labels)
        if not merge_candidates:
            break
        best_merge = min(merge_candidates, key=lambda x: x[4])
        (pair, fused_labels, fused_sil, fused_disp_dict, fused_disp_sum) = best_merge
        unique_after_merge = np.unique(fused_labels)
        if len(unique_after_merge) == 2:
            valid_merge = True
            for c in unique_after_merge:
                n_atoms = np.sum(fused_labels == c)
                if n_atoms < min_atoms:
                    valid_merge = False
                    print(f"  > La fusión {pair} produce el clúster {c} con solo {n_atoms} átomos (<{min_atoms}).")
                    break
            if not valid_merge:
                iteration += 1
                continue
        print(f"  Fusión elegida: {pair}, disp_total={fused_disp_sum:.3f}, sil={fused_sil:.3f}")
        labels = fused_labels

        worst_cluster, max_disp = get_worst_cluster(fused_disp_dict)
        print(f"  Clúster con mayor dispersión: {worst_cluster}, valor={max_disp:.3f}")
        mask = (labels == worst_cluster)
        n_atoms = np.sum(mask)
        if max_disp > threshold:
            if n_atoms < min_atoms:
                print(f"  > El clúster {worst_cluster} tiene solo {n_atoms} átomos (<{min_atoms}). No se subdivide.")
            else:
                print(f"  > Dispersión > {threshold} y {n_atoms} átomos. Intentando subdividir clúster {worst_cluster} con KMeans(3)...")
                coords_worst = coords[mask]
                # Dividimos en 3 subclusters
                sub_labels, sub_sil, sub_disp_dict, sub_disp_sum = kmeans_three_points(coords_worst)
                unique_sub = np.unique(sub_labels)
                # Ahora, intentamos fusionar dos de los tres subclusters para obtener dos clusters finales.
                candidate_fusions = []
                for i in range(len(unique_sub)):
                    for j in range(i+1, len(unique_sub)):
                        c1 = unique_sub[i]
                        c2 = unique_sub[j]
                        fused_sub = merge_clusters(sub_labels, c1, c2)
                        fused_unique = np.unique(fused_sub)
                        if len(fused_unique) != 2:
                            continue
                        counts = [np.sum(fused_sub == lab) for lab in fused_unique]
                        candidate_fusions.append(((c1, c2), fused_sub, counts))
                valid_fusion = None
                for candidate in candidate_fusions:
                    (fusion_pair, fused_sub, counts) = candidate
                    if all(count >= min_atoms for count in counts):
                        valid_fusion = candidate
                        break
                if valid_fusion is None:
                    print(f"  > La subdivisión del clúster {worst_cluster} produce dos clusters finales con menos de {min_atoms} átomos. Se revierte la subdivisión.")
                else:
                    (fusion_pair, fused_sub, counts) = valid_fusion
                    print(f"  > Subdivisión aceptada: se fusionan los subclusters {fusion_pair} resultando en dos clusters con conteos {counts}.")
                    # Asignamos estos nuevos labels al clúster worst_cluster en el global.
                    offset = worst_cluster * 10
                    new_sub_labels = offset + fused_sub
                    new_labels_global = np.copy(labels)
                    new_labels_global[mask] = new_sub_labels
                    labels = new_labels_global
        else:
            print(f"  > Dispersión <= {threshold}. No se subdivide.")
        iteration += 1
    print(f"\n[FIN] Iteraciones completadas = {iteration}. Clústeres finales: {np.unique(labels)}")
    return labels


class ClusterProcessorMachine:
    def __init__(self, file_path: str, threshold: float = 1.2, max_iterations: int = 10, min_atoms: int = None):
        self.file_path = file_path
        self.threshold = threshold
        self.max_iterations = max_iterations
        if min_atoms is None:
            self.min_atoms = UtilidadesClustering.cargar_min_atoms("outputs.vfinder/key_single_vacancy.json")
        else:
            self.min_atoms = min_atoms
        self.matriz_total = UtilidadesClustering.extraer_datos_completos(file_path)
        self.header = UtilidadesClustering.extraer_encabezado(file_path)
        pipeline = import_file(file_path)
        data = pipeline.compute()
        self.coords = data.particles.positions
        self.init_labels = data.particles["Cluster"].array
    
    def process_clusters(self):
        self.final_labels = iterative_fusion_and_subdivision(self.coords, self.init_labels, self.threshold, self.max_iterations, self.min_atoms)
        #print("\nClústeres finales en 'final_labels':", np.unique(self.final_labels))
        unique = np.unique(self.final_labels)
        mapping = {old: new for new, old in enumerate(unique)}
        self.final_labels = np.vectorize(mapping.get)(self.final_labels)
        #print("Clústeres remapeados:", np.unique(self.final_labels))
        if self.matriz_total.shape[0] == self.final_labels.shape[0]:
            self.matriz_total[:, 5] = self.final_labels
        else:
            print("¡Atención! La cantidad de filas en la matriz de datos y los clusters no coincide.")
    
    def export_updated_file(self, output_file: str = None):
        if output_file is None:
            output_file = f"{self.file_path}"
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.writelines(self.header)
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
            #print("Datos exportados exitosamente a:", output_file)
        except Exception as e:
            print(f"Error al exportar {output_file}: {e}")

class ExportClusterList:
    def __init__(self, json_path="outputs.json/key_archivos.json"):
        self.json_path = json_path
        self.load_config()
    
    def load_config(self):
        with open(self.json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.clusters_criticos = self.data.get("clusters_criticos", [])
        self.clusters_final = self.data.get("clusters_final", [])
    
    def save_config(self):
        self.data["clusters_final"] = self.clusters_final
        with open(self.json_path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, indent=4)
    
    def obtener_grupos_cluster(self, file_path):
        clusters = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
        atom_header_line = None
        for i, line in enumerate(lines):
            if line.startswith("ITEM: ATOMS"):
                atom_header_line = line.strip()
                data_start = i + 1
                break
        if atom_header_line is None:
            raise ValueError("No se encontró la sección 'ITEM: ATOMS' en el archivo.")
        header_parts = atom_header_line.split()[2:]
        try:
            cluster_index = header_parts.index("Cluster")
        except ValueError:
            raise ValueError("La columna 'Cluster' no se encontró en la cabecera.")
        for line in lines[data_start:]:
            if line.startswith("ITEM:"):
                break
            if line.strip() == "":
                continue
            parts = line.strip().split()
            if len(parts) <= cluster_index:
                continue
            clusters.append(parts[cluster_index])
        unique_clusters = set(clusters)
        return unique_clusters, clusters

    def process_files(self):
        for archivo in self.clusters_criticos:
            try:
                unique_clusters, _ = self.obtener_grupos_cluster(archivo)
            except Exception as e:
                continue
            for i in range(0, len(unique_clusters)):
                pipeline = import_file(archivo)
                pipeline.modifiers.append(ExpressionSelectionModifier(expression=f"Cluster!={i}"))
                pipeline.modifiers.append(DeleteSelectedModifier())
                try:
                    nuevo_archivo = f"{archivo}.{i}"
                    export_file(pipeline, nuevo_archivo, "lammps/dump", 
                                columns=["Particle Identifier", "Particle Type", "Position.X", "Position.Y", "Position.Z", "Cluster"])
                    pipeline.modifiers.clear()
                    self.clusters_final.append(nuevo_archivo)
                except Exception as e:
                    pass
        self.save_config()

class KeyFilesSeparator:
    def __init__(self, config, clusters_json_path):
        self.config = config
        self.cluster_tolerance = config.get("cluster tolerance", 1.7)
        self.clusters_json_path = clusters_json_path
        self.lista_clusters_final = []
        self.lista_clusters_criticos = []
        self.num_clusters = self.cargar_num_clusters()

    def cargar_num_clusters(self):
        if not os.path.exists(self.clusters_json_path):
            return 0
        with open(self.clusters_json_path, "r", encoding="utf-8") as f:
            datos = json.load(f)
        num = datos.get("num_clusters", 0)
        return num

    def extraer_coordenadas(self, file_path):
        coordenadas = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except FileNotFoundError:
            return coordenadas
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break
        if start_index is None:
            return coordenadas
        for line in lines[start_index:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            try:
                x = float(parts[2])
                y = float(parts[3])
                z = float(parts[4])
                coordenadas.append((x, y, z))
            except ValueError:
                continue
        return coordenadas

    def calcular_centro_de_masa(self, coordenadas):
        arr = np.array(coordenadas)
        if arr.size == 0:
            return None
        centro = arr.mean(axis=0)
        return tuple(centro)

    def calcular_dispersion(self, coordenadas, centro_de_masa):
        if coordenadas is None or (hasattr(coordenadas, '__len__') and len(coordenadas) == 0) or centro_de_masa is None:
            return [], 0
        distancias = []
        cx, cy, cz = centro_de_masa
        for (x, y, z) in coordenadas:
            d = math.sqrt((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2)
            distancias.append(d)
        dispersion = np.std(distancias)
        return distancias, dispersion

    def construir_matriz_coordenadas(self, archivo):
        coords = self.extraer_coordenadas(archivo)
        matriz = []
        for (x, y, z) in coords:
            matriz.append([x, y, z, 0])
        return np.array(matriz)

    def separar_archivos(self):
        for i in range(1, self.num_clusters + 1):
            ruta_archivo = f"outputs.dump/key_area_{i}.dump"
            coords = self.extraer_coordenadas(ruta_archivo)
            centroide = self.calcular_centro_de_masa(coords)
            distancias, dispersion = self.calcular_dispersion(coords, centroide)
            if dispersion > self.cluster_tolerance:
                self.lista_clusters_criticos.append(ruta_archivo)
            else:
                self.lista_clusters_final.append(ruta_archivo)

    def exportar_listas(self, output_path):
        datos_exportar = {
            "clusters_criticos": self.lista_clusters_criticos,
            "clusters_final": self.lista_clusters_final
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(datos_exportar, f, indent=4)

    def run(self):
        self.separar_archivos()
        self.exportar_listas("outputs.json/key_archivos.json")
import os
import json
import math
import numpy as np
import pandas as pd
import xgboost as xgb
from ovito.io import import_file, export_file
from ovito.modifiers import DeleteSelectedModifier, InvertSelectionModifier, ExpressionSelectionModifier
from input_params import CONFIG
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import json
import os
import csv
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans

class CDScanner:
    def __init__(self):
        self.config = CONFIG[0]
        with open('outputs.json/key_archivos.json', 'r', encoding='utf-8') as f:
            clusters_data = json.load(f)
        self.clusters_final = clusters_data.get("clusters_final", [])
        self.lista_centros = []
        self.lista_num_atomos = []
        self.centros_np = None
        self.etiquetas = None
        self.centros_kmeans = None
        self.df_clusters = None
        self.new_header = None
        self.silhouette_by_cluster = None

    def calcular_centroide(self, positions):
        return positions.mean(axis=0)

    def process_clusters(self):
        for cluster_file in self.clusters_final:
            pipeline_cluster = import_file(cluster_file)
            data_cluster = pipeline_cluster.compute()
            positions = data_cluster.particles.position.array
            if positions.size == 0:
                continue
            centroide = self.calcular_centroide(positions)
            num_atomos = data_cluster.particles.count
            self.lista_centros.append(centroide)
            self.lista_num_atomos.append(num_atomos)
        if len(self.lista_centros) == 0:
            raise Exception("No se han obtenido centros de masa.")
        self.centros_np = np.array(self.lista_centros)

    def run_kmeans(self):
        num_clusters = self.config.get("k_means_clusters", 3)
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(self.centros_np)
        self.etiquetas = kmeans.labels_
        self.centros_kmeans = kmeans.cluster_centers_
        inercia = []
        k_range = range(1, len(self.centros_np) + 1)
        for k in k_range:
            kmeans_temp = KMeans(n_clusters=k, init='k-means++', random_state=0)
            kmeans_temp.fit(self.centros_np)
            inercia.append(kmeans_temp.inertia_)
        try:
            from kneed import KneeLocator
            kneedle = KneeLocator(list(k_range), inercia, curve="convex", direction="decreasing")
            k_optimo = kneedle.knee
            if k_optimo is None:
                k_optimo = num_clusters
        except ImportError:
            k_optimo = num_clusters
        kmeans_final = KMeans(n_clusters=k_optimo, init='k-means++', random_state=0)
        kmeans_final.fit(self.centros_np)
        self.etiquetas = kmeans_final.labels_
        self.centros_kmeans = kmeans_final.cluster_centers_

    def create_dataframe(self):
        self.df_clusters = pd.DataFrame({
            'id': np.arange(len(self.centros_np)),
            'x': self.centros_np[:, 0],
            'y': self.centros_np[:, 1],
            'z': self.centros_np[:, 2],
            'Cluster': self.etiquetas,
            'NumAtomos': self.lista_num_atomos
        })
        self.df_clusters['type'] = 1
        self.df_clusters = self.df_clusters[['id', 'type', 'x', 'y', 'z', 'Cluster', 'NumAtomos']]

    def update_dump_header(self):
        header_lines = []
        with open("outputs.dump/key_areas.dump", "r") as f:
            for line in f:
                header_lines.append(line.rstrip("\n"))
                if line.startswith("ITEM: ATOMS"):
                    break
        n_atoms = len(self.df_clusters)
        new_header = []
        i = 0
        while i < len(header_lines):
            line = header_lines[i]
            if line.startswith("ITEM: NUMBER OF ATOMS"):
                new_header.append(line)
                new_header.append(str(n_atoms))
                i += 2
                continue
            else:
                new_header.append(line)
            i += 1
        for idx, line in enumerate(new_header):
            if line.startswith("ITEM: ATOMS"):
                new_header[idx] = "ITEM: ATOMS id type x y z etiqueta"
                break
        self.new_header = new_header

    def export_dump_file(self):
        with open("outputs.dump/city_population.dump", "w") as f:
            for line in self.new_header:
                f.write(line + "\n")
            self.df_clusters.to_csv(f, sep=" ", index=False, header=False)

    def calculate_silhouette(self):
        from sklearn.metrics import silhouette_samples
        silhouette_vals = silhouette_samples(self.centros_np, self.etiquetas)
        self.df_clusters['Silhouette'] = silhouette_vals
        self.silhouette_by_cluster = self.df_clusters.groupby('Cluster')['Silhouette'].mean()

    def export_mapping(self):
        mapping_df = pd.DataFrame({
                'Archivo': self.clusters_final,
                'Cluster': self.df_clusters['Cluster']
            })
        output_folder = "outputs.csv"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        mapping_df.to_csv(os.path.join(output_folder, "mapeo_archivos_cluster.csv"), index=False)

    def export_figures(self):
        output_folder = "outputs.csv"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        defect_path = self.config.get('defect', 'default_defect')
        if isinstance(defect_path, list):
            defect_path = defect_path[0]
        defect_name = os.path.basename(defect_path)
        import scipy.stats as stats
        x = self.df_clusters['x']
        y = self.df_clusters['y']
        z = self.df_clusters['z']
        positions = np.vstack([x, y, z])
        density = stats.gaussian_kde(positions)(positions)
        fig_3d = plt.figure(figsize=(10,8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        sc = ax_3d.scatter(x, y, z, c=density, cmap='inferno', s=50, edgecolor='black')
        fig_3d.colorbar(sc, ax=ax_3d, label='Densidad')
        ax_3d.set_title("(densidad de puntos)")
        ax_3d.set_xlabel("x[Å]")
        ax_3d.set_ylabel("y[Å]")
        ax_3d.set_zlabel("z[Å]")
        output_file_3d = os.path.join(output_folder, f"{defect_name}_3D_heatmap.png")
        fig_3d.savefig(output_file_3d)
        plt.close(fig_3d)
        x_arr = self.df_clusters['x'].values
        y_arr = self.df_clusters['y'].values
        z_arr = self.df_clusters['z'].values
        bins = 50
        xy_hist, xedges, yedges = np.histogram2d(x_arr, y_arr, bins=bins)
        extent_xy = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        yz_hist, yedges2, zedges = np.histogram2d(y_arr, z_arr, bins=bins)
        extent_yz = [yedges2[0], yedges2[-1], zedges[0], zedges[-1]]
        zx_hist, zedges2, xedges2 = np.histogram2d(z_arr, x_arr, bins=bins)
        extent_zx = [zedges2[0], zedges2[-1], xedges2[0], xedges2[-1]]
        fig_contour, axs = plt.subplots(1, 3, figsize=(18,5))
        cset_xy = axs[0].contourf(xy_hist.T, levels=20, extent=extent_xy, cmap='viridis')
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].grid(True)
        fig_contour.colorbar(cset_xy, ax=axs[0])
        cset_yz = axs[1].contourf(yz_hist.T, levels=20, extent=extent_yz, cmap='viridis')
        axs[1].set_xlabel("y")
        axs[1].set_ylabel("z")
        axs[1].grid(True)
        fig_contour.colorbar(cset_yz, ax=axs[1])
        cset_zx = axs[2].contourf(zx_hist.T, levels=20, extent=extent_zx, cmap='viridis')
        axs[2].set_xlabel("z")
        axs[2].set_ylabel("x")
        axs[2].grid(True)
        fig_contour.colorbar(cset_zx, ax=axs[2])
        plt.tight_layout()
        output_file_contours = os.path.join(output_folder, f"{defect_name}_contour_maps.png")
        fig_contour.savefig(output_file_contours)
        plt.close(fig_contour)
        poblacion_por_cluster = self.df_clusters.groupby('Cluster')['NumAtomos'].sum()
        poblacion_por_cluster_sorted = poblacion_por_cluster.sort_values(ascending=False)
        plt.figure(figsize=(12,6))
        poblacion_por_cluster_sorted.plot(kind='bar', color='skyblue')
        plt.xlabel("Cluster ID")
        plt.ylabel("Número total de átomos")
        plt.grid(True)
        plt.tight_layout()
        bar_plot_file = os.path.join(output_folder, f"{defect_name}_pop_cluster_bar.png")
        plt.savefig(bar_plot_file)
        plt.close()
        import seaborn as sns
        df_pop = poblacion_por_cluster_sorted.reset_index()
        df_pop.columns = ['Cluster', 'NumAtomos']
        plt.figure(figsize=(12,2))
        sns_heatmap = sns.heatmap(df_pop[['NumAtomos']].T, annot=True, fmt="d", cmap='viridis', cbar=True)
        plt.xlabel("Cluster")
        plt.ylabel("Número total de átomos")
        plt.tight_layout()
        heatmap_file = os.path.join(output_folder, f"{defect_name}_pop_cluster_heatmap.png")
        plt.savefig(heatmap_file)
        plt.close()





import csv
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os

class HistogramaArchivoCSV:
    def __init__(self, ruta):
        self.ruta = ruta
        self.nombres_archivos = []
        self.clusters = []

    def leer_archivo(self):
        with open(self.ruta, mode='r', newline='', encoding='utf-8') as archivo:
            lector_csv = csv.DictReader(archivo)
            for fila in lector_csv:
                self.nombres_archivos.append(fila['Archivo'])
                self.clusters.append(int(fila['Cluster']))

class HistogramaFile:
    def __init__(self, nombres_archivos):
        self.nombres_archivos = nombres_archivos
        self.resultados = []

    def extraer_tipo_columna(self, archivo):
        tipo_columna = []
        read_data = False
        with open(archivo, 'r') as file:
            for line in file:
                line = line.strip()
                if line.startswith("ITEM: ATOMS"):
                    read_data = True
                    continue
                if read_data:
                    columns = line.split()
                    if len(columns) >= 2:
                        tipo_columna.append(columns[1])
        return tipo_columna

    def procesar_archivos(self, clusters):
        for i, archivo in enumerate(self.nombres_archivos):
            tipo_valores = self.extraer_tipo_columna(archivo)
            conteo_tipos = Counter(tipo_valores)
            self.resultados.append({
                'Archivo': archivo,
                'Cluster': clusters[i],
                'Conteo_Tipos': dict(conteo_tipos)
            })

class HistogramaTypes:
    def __init__(self, resultados):
        self.resultados = resultados

    def obtener_tipos_unicos(self):
        tipos_unicos = set()
        for resultado in self.resultados:
            tipos_unicos.update(resultado['Conteo_Tipos'].keys())
        return sorted(tipos_unicos, key=lambda x: int(x))

    def crear_matriz_frecuencias(self, clusters_unicos, tipos_unicos):
        matriz_frecuencias = []
        for cluster in clusters_unicos:
            frecuencias_cluster = {tipo: 0 for tipo in tipos_unicos}
            for resultado in self.resultados:
                if resultado['Cluster'] == cluster:
                    for tipo, frecuencia in resultado['Conteo_Tipos'].items():
                        frecuencias_cluster[tipo] += frecuencia
            fila = [cluster] + [frecuencias_cluster[tipo] for tipo in tipos_unicos]
            matriz_frecuencias.append(fila)
        return matriz_frecuencias

    def graficar(self, matriz_frecuencias, tipos_unicos, nombre_archivo):
        suma_frecuencias = [sum(fila[1:]) for fila in matriz_frecuencias]
        indices_ordenados = np.argsort(suma_frecuencias)[::-1]
        matriz_frecuencias = [matriz_frecuencias[i] for i in indices_ordenados]
        clusters_ordenados = [fila[0] for fila in matriz_frecuencias]
        x_positions = np.arange(len(clusters_ordenados))
        frecuencias = np.array([fila[1:] for fila in matriz_frecuencias])
        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(clusters_ordenados))
        for i, tipo in enumerate(tipos_unicos):
            ax.bar(x_positions, frecuencias[:, i], bottom=bottom, label=f'Tipo {tipo}')
            bottom += frecuencias[:, i]
        for j, pos in enumerate(x_positions):
            total = np.sum(frecuencias[j])
            if total > 0:
                acumulado = 0
                for i, valor in enumerate(frecuencias[j]):
                    if valor > 0:
                        porcentaje = (valor / total) * 100
                        y_pos = acumulado + valor / 2
                        ax.text(pos, y_pos, f'{porcentaje:.1f}%', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
                    acumulado += valor
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Frecuencia de átomos')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(clusters_ordenados)
        ax.legend(title='Tipos de átomos', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        nombre_archivo_sin_extension = os.path.splitext(os.path.basename(nombre_archivo))[0]
        plt.savefig(f'{nombre_archivo_sin_extension}.png')
        plt.show()

if __name__ == "__main__":
    config = CONFIG[0]
    relax = config['relax']
    radius_training = config['radius_training']
    radius = config['radius']
    smoothing_level_training = config['smoothing_level_training']
    other_method = config['other method']
    
    processor = ClusterProcessor()
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


    scanner = CDScanner()
    scanner.process_clusters()
    scanner.run_kmeans()
    scanner.create_dataframe()
    scanner.update_dump_header()
    scanner.export_dump_file()
    scanner.calculate_silhouette()
    scanner.export_mapping()
    scanner.export_figures()
   

           


