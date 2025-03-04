import numpy as np
from sklearn.cluster import KMeans

def extract_atoms_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    atoms_data = []
    reading_atoms = False

    for line in lines:
        line = line.strip()
        if line.startswith("ITEM: ATOMS"):
            reading_atoms = True
            continue
        if reading_atoms and line.startswith("ITEM:"):
            break
        if reading_atoms and line:
            try:
                row = [float(x) for x in line.split()]
                atoms_data.append(row)
            except ValueError:
                continue

    if atoms_data:
        data_matrix = np.array(atoms_data)
    else:
        data_matrix = np.empty((0, 6))
    return data_matrix

def extract_coordinates_with_cluster(data_matrix):
    if data_matrix.shape[1] < 6:
        print("La matriz de datos no tiene suficientes columnas.")
        return None
    coords_cluster_matrix = data_matrix[:, 2:6]
    return coords_cluster_matrix

def compute_center_and_farthest_points(coords):
    if coords.shape[0] == 0:
        return None, None, None

    center_of_mass = np.mean(coords, axis=0)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.linalg.norm(diff, axis=2)
    i, j = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    max_distance = dist_matrix[i, j]
    farthest_points = coords[[i, j], :]
    return center_of_mass, farthest_points, max_distance

def apply_kmeans_with_initial_points(coords, initial_points):
    kmeans = KMeans(n_clusters=3, init=initial_points, n_init=1, random_state=42)
    kmeans.fit(coords)
    return kmeans.labels_, kmeans.cluster_centers_

def update_cluster_labels(coords_cluster_matrix, new_labels):
    updated_matrix = np.copy(coords_cluster_matrix)
    updated_matrix[:, 3] = new_labels
    return updated_matrix

def split_by_cluster(coords_cluster_matrix):
    clusters_dict = {}
    unique_clusters = np.unique(coords_cluster_matrix[:, 3])
    for cluster in unique_clusters:
        mask = coords_cluster_matrix[:, 3] == cluster
        clusters_dict[cluster] = coords_cluster_matrix[mask, :]
    return clusters_dict

def compute_cluster_stats(clusters_dict):
    """
    Dado un diccionario de clusters, donde cada valor es una matriz con columnas (x, y, z, Cluster),
    calcula para cada cluster:
      - El centro de masa: media de las coordenadas x, y, z.
      - La dispersión: desviación estándar de las distancias de cada punto al centro de masa.
    
    Retorna un diccionario de la forma:
      { cluster_label: (center_of_mass, dispersion) }
    """
    stats = {}
    for label, submatrix in clusters_dict.items():
        coords = submatrix[:, 0:3]
        center = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        dispersion = np.std(distances)
        stats[label] = (center, dispersion)
    return stats

def main():
    file_path = "outputs.dump/key_area_1.dump"
    
    data_matrix = extract_atoms_data(file_path)
    if data_matrix.size == 0:
        print("No se encontraron datos de átomos en el archivo.")
        return
    else:
        print("Matriz completa de datos extraída:")
        print("Dimensiones:", data_matrix.shape)
        print(data_matrix)
        
        coords_cluster_matrix = extract_coordinates_with_cluster(data_matrix)
        if coords_cluster_matrix is not None:
            print("\nMatriz de coordenadas y Cluster (x, y, z, Cluster):")
            print("Dimensiones:", coords_cluster_matrix.shape)
            print(coords_cluster_matrix)
            
            coords = coords_cluster_matrix[:, 0:3]
            center, farthest_pts, max_dist = compute_center_and_farthest_points(coords)
            
            print("\nCentro de masa (x, y, z):")
            print(center)
            print("\nPuntos con mayor distancia:")
            print(f"Punto 1: {farthest_pts[0]}")
            print(f"Punto 2: {farthest_pts[1]}")
            print(f"Distancia máxima: {max_dist}")
            
            initial_points = np.vstack([center, farthest_pts[0], farthest_pts[1]])
            print("\nPuntos iniciales para KMeans:")
            print(initial_points)
            
            labels, cluster_centers = apply_kmeans_with_initial_points(coords, initial_points)
            print("\nEtiquetas de cada punto:")
            print(labels)
            print("\nCentros de los clusters obtenidos:")
            print(cluster_centers)
            
            updated_coords_cluster_matrix = update_cluster_labels(coords_cluster_matrix, labels)
            print("\nMatriz de coordenadas actualizada (con nueva columna Cluster):")
            print("Dimensiones:", updated_coords_cluster_matrix.shape)
            print(updated_coords_cluster_matrix)
            
            clusters_dict = split_by_cluster(updated_coords_cluster_matrix)
            print("\nDivisión de coordenadas según el cluster:")
            for cluster, submatrix in clusters_dict.items():
                print(f"\nCluster {int(cluster)}:")
                print(submatrix)
            
            # Calcular centro de masa y dispersión para cada cluster
            stats = compute_cluster_stats(clusters_dict)
            print("\nCentro de masa y dispersión por cluster:")
            for cluster, (center, dispersion) in stats.items():
                print(f"Cluster {int(cluster)}: Centro de masa = {center}, Dispersion = {dispersion}")

if __name__ == '__main__':
    main()
