import os
import json
import numpy as np
from ovito.io import import_file
from ovito.modifiers import ConstructSurfaceModifier

class ClusterDumpProcessor:
    def __init__(self, file_path: str, decimals: int = 5, json_params_path: str = "input_params.json"):
        
        self.file_path = file_path
        self.matriz_total = None
        self.header = None
        self.subset = None

        # 1) Cargar input_params.json
        try:
            with open(json_params_path, "r", encoding="utf-8") as f:
                all_params = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de parámetros: {json_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear {json_params_path}: {e}")

        # 2) Extraer la primera entrada de CONFIG
        if "CONFIG" not in all_params or not isinstance(all_params["CONFIG"], list) or len(all_params["CONFIG"]) == 0:
            raise KeyError("El JSON no contiene una lista válida en la clave 'CONFIG'.")
        self.config = all_params["CONFIG"][0]

        # 3) Tomar 'divisions_of_cluster' del JSON
        try:
            self.divisions_of_cluster = self.config["divisions_of_cluster"]
        except KeyError:
            raise KeyError("Falta la clave 'divisions_of_cluster' en el JSON de CONFIG.")

        # 4) Decimales para redondeo en coordenadas
        self.decimals = decimals


    def load_data(self):
        """
        Extrae la matriz completa de datos (id, type, x, y, z, cluster)
        y el encabezado hasta 'ITEM: ATOMS'. Luego arma self.subset = columnas [x,y,z].
        """
        self.matriz_total = self.extraer_datos_completos(self.file_path)
        self.header = self.extraer_encabezado(self.file_path)
        if self.matriz_total.size == 0:
            raise ValueError(f"No se pudieron extraer datos de {self.file_path}")
        # Columnas 2,3,4 son x,y,z
        self.subset = self.matriz_total[:, 2:5]


    def calcular_dispersion(self, points: np.ndarray) -> float:
        """
        Calcula la distancia promedio de cada punto al centro de masa.
        """
        if points.shape[0] == 0:
            return 0.0
        center = np.mean(points, axis=0)
        distances = np.linalg.norm(points - center, axis=1)
        return np.mean(distances)


    def process_clusters(self):
        """
        Toma self.subset (x,y,z de todos los átomos), halla dos puntos más alejados
        y el centro de masa global. Luego aplica k-means con inicialización a 3 centros:
          [p1, p2, centro_masa_global].
        Finalmente, sobrescribe la columna 5 de self.matriz_total con las etiquetas
        resultantes (0,1 o 2).
        """
        if self.subset is None:
            raise ValueError("Los datos no han sido cargados. Ejecuta load_data() primero.")

        # 1) Centro de masa global
        centro_masa_global = np.mean(self.subset, axis=0)

        # 2) Dos puntos más alejados y distancia máxima
        p1, p2, distancia_maxima = self.find_farthest_points(self.subset)

        # 3) (Opcional) Podrías usar self.calcular_dispersion(self.subset) si quisieras algo con la dispersión.
        dispersion = self.calcular_dispersion(self.subset)

        # 4) Umbral de división (no se usa explícitamente aquí, pero está disponible)
        threshold = self.divisions_of_cluster

        # 5) Aplicar k-means para 3 clusters
        etiquetas = self.aplicar_kmeans(self.subset, p1, p2, centro_masa_global, n_clusters=3)

        # 6) Verificar que la cantidad de etiquetas coincide con filas totales
        if etiquetas.shape[0] != self.matriz_total.shape[0]:
            raise ValueError("El número de etiquetas no coincide con la matriz total.")

        # 7) Sobrescribir la columna 5 (índice 5) con las nuevas etiquetas
        self.matriz_total[:, 5] = etiquetas


    def export_updated_file(self, output_file: str = None):
        """
        Guarda en disco la matriz total (con la columna de cluster actualizada).
        Si no se pasa output_file, genera uno con sufijo "_actualizado.txt".
        """
        if output_file is None:
            output_file = f"{self.file_path}_actualizado.txt"

        # Formato del archivo: id, type, x, y, z, cluster (enteros/floats con 5 decimales)
        fmt = ("%d %d %.5f %.5f %.5f %d")
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                # Escribimos el encabezado original (líneas hasta ITEM: ATOMS)
                f.writelines(self.header)
                # Luego, la matriz con la columna de cluster actualizada
                np.savetxt(f, self.matriz_total, fmt=fmt, delimiter=" ")
        except Exception as e:
            raise IOError(f"Error al escribir {output_file}: {e}")


    @staticmethod
    def cargar_lista_archivos_criticos(json_path: str) -> list:
        """
        Lee un JSON (p.ej. outputs.json/key_archivos.json) y retorna la lista
        bajo la clave "clusters_criticos". Si no existe o no es válido, devuelve [].
        """
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                datos = json.load(f)
            return datos.get("clusters_criticos", [])
        except FileNotFoundError:
            return []
        except json.JSONDecodeError:
            return []


    @staticmethod
    def extraer_datos_completos(file_path: str) -> np.ndarray:
        """
        Abre un dump de LAMMPS (formato texto), busca la línea 'ITEM: ATOMS', y luego
        lee cada línea que siga como: id, type, x, y, z, cluster.
        Redondea x,y,z a 5 decimales.
        Retorna un array NumPy de forma (N, 6).
        """
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
                # Si algo falla en la conversión, ignoramos esa línea
                continue

        return np.array(datos)


    @staticmethod
    def extraer_encabezado(file_path: str) -> list:
        """
        Lee el archivo completo hasta encontrar 'ITEM: ATOMS'. Retorna una lista
        de líneas (incluyendo 'ITEM: ATOMS') para usar luego en export_updated_file.
        """
        encabezado = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    encabezado.append(line)
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except Exception:
            # Si ocurre cualquier error, devolvemos una lista vacía
            pass
        return encabezado


    @staticmethod
    def aplicar_kmeans(coordenadas: np.ndarray, p1, p2, centro_masa_global, n_clusters: int) -> np.ndarray:
        """
        Inicializa KMeans con n_clusters==2 o n_clusters==3 usando centros dados.
        Si n_clusters == 2, usa [p1, p2], si es 3, usa [p1, p2, centro_masa_global].
        Retorna el array de etiquetas (shape = (N,)).
        """
        from sklearn.cluster import KMeans

        if n_clusters == 2:
            init_centers = np.array([p1, p2])
        elif n_clusters == 3:
            init_centers = np.array([p1, p2, centro_masa_global])
        else:
            raise ValueError("Solo se admite n_clusters igual a 2 o 3.")

        kmeans = KMeans(
            n_clusters=n_clusters,
            init=init_centers,
            n_init=1,
            max_iter=300,
            tol=1e-4,
            random_state=42
        )
        etiquetas = kmeans.fit_predict(coordenadas)
        return etiquetas


    @staticmethod
    def find_farthest_points(coordenadas: np.ndarray):
        """
        Dada una matriz (N,3), halla el par de puntos más alejados entre sí:
        - pts: array de shape (N,3)
        - diffs: shape (N,N,3) con diferencias entre todos los pares
        - distancias: matriz (N,N) de distancias euclídeas
        Retorna: (punto1, punto2, distancia_maxima)
        """
        pts = np.array(coordenadas)
        n = pts.shape[0]
        if n < 2:
            return None, None, 0

        # Creamos todas las diferencias de pares de puntos
        diffs = pts[:, None, :] - pts[None, :, :]
        distancias = np.sqrt(np.sum(diffs**2, axis=-1))
        idx = np.unravel_index(np.argmax(distancias), distancias.shape)
        distancia_maxima = distancias[idx]
        punto1 = pts[idx[0]]
        punto2 = pts[idx[1]]
        return punto1, punto2, distancia_maxima
