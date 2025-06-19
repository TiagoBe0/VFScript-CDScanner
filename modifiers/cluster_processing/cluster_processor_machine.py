
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
        self.min_atoms = UtilidadesClustering.cargar_min_atoms("outputs.vfinder/key_single_vacancy.json")

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
