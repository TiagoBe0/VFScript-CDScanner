import os
import json
import numpy as np
import pandas as pd
from ovito.io import import_file
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

class CDScanner:
    def __init__(
        self,
        json_params_path: str = "input_params.json",
        key_archivos_path: str = "outputs.json/key_archivos.json"
    ):
        """
        Ahora la configuración (por ejemplo, 'defect' o 'k_means_clusters') se extrae
        directamente de input_params.json en lugar de importarla de un módulo .py.
        Se espera que input_params.json contenga al menos:
        {
          "CONFIG": [
            {
              "defect": [...],
              "radius": <número o lista>,
              "smoothing_level": <entero>,
              "cutoff": <número>,
              "cluster tolerance": <número>,
              "divisions_of_cluster": <entero>,
              "iteraciones_clusterig": <entero>,
              ...otras claves opcionales...
            }
          ],
          "PREDICTOR_COLUMNS": [...]
        }
        """
        # 1) Cargar input_params.json
        try:
            with open(json_params_path, "r", encoding="utf-8") as f:
                all_params = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de parámetros: {json_params_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear {json_params_path}: {e}")

        if "CONFIG" not in all_params or not isinstance(all_params["CONFIG"], list) or len(all_params["CONFIG"]) == 0:
            raise KeyError("El JSON no contiene una lista válida en la clave 'CONFIG'.")

        self.config = all_params["CONFIG"][0]

        # 2) Obtener 'defect' (puede ser lista o string)
        defect_val = self.config.get("defect", None)
        if defect_val is None:
            raise KeyError("Falta la clave 'defect' en el JSON de CONFIG.")
        # Si viene como lista, tomamos el primer elemento
        if isinstance(defect_val, list):
            self.defect = defect_val[0]
        else:
            self.defect = defect_val

        # 3) Cargar la lista de dumps finales desde outputs.json/key_archivos.json
        try:
            with open(key_archivos_path, "r", encoding="utf-8") as f:
                clusters_data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el archivo de clusters: {key_archivos_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear {key_archivos_path}: {e}")

        self.clusters_final = clusters_data.get("clusters_final", [])
        if not isinstance(self.clusters_final, list):
            raise ValueError(f"'clusters_final' debe ser una lista en {key_archivos_path}")

        # 4) Parámetros para k-means
        #    Permitimos que el usuario defina 'k_means_clusters' en el JSON; si no, usamos 3
        self.k_means_clusters = self.config.get("k_means_clusters", 3)

        # 5) Listas para almacenar resultados
        self.lista_centros = []
        self.lista_num_atomos = []
        self.centros_np = None
        self.etiquetas = None
        self.centros_kmeans = None
        self.df_clusters = None
        self.new_header = None
        self.silhouette_by_cluster = None


    def calcular_centroide(self, positions: np.ndarray) -> np.ndarray:
        """
        Dado un array (N,3) de posiciones, devuelve su centroide.
        """
        return positions.mean(axis=0)


    def process_clusters(self):
        """
        Itera sobre cada dump en self.clusters_final, calcula su centroide y
        cuenta de átomos, y almacena estos datos en listas intermedias.
        Luego construye self.centros_np = array de centroides.
        """
        for cluster_file in self.clusters_final:
            pipeline_cluster = import_file(cluster_file)
            data_cluster = pipeline_cluster.compute()

            # Acceder a las posiciones: 
            # data_cluster.particles.position.array en lugar de .particles.position.array
            positions = data_cluster.particles.position.array
            if positions.size == 0:
                # Si no hay partículas, pasamos al siguiente
                continue

            centroide = self.calcular_centroide(positions)
            num_atomos = data_cluster.particles.count

            self.lista_centros.append(centroide)
            self.lista_num_atomos.append(num_atomos)

        if len(self.lista_centros) == 0:
            raise Exception("No se han obtenido centros de masa de ningún cluster.")

        self.centros_np = np.vstack(self.lista_centros)


    def run_kmeans(self):
        """
        Ejecuta k-means inicial con el número de clusters definido por 'k_means_clusters',
        guarda etiquetas e inercia. Luego, si la librería kneed está disponible, busca
        la rodilla óptima y refina el clustering.
        """
        # 1) Clustering inicial con k = self.k_means_clusters
        kmeans = KMeans(n_clusters=self.k_means_clusters, random_state=0)
        kmeans.fit(self.centros_np)
        self.etiquetas = kmeans.labels_
        self.centros_kmeans = kmeans.cluster_centers_

        # 2) Calcular inercia para distintos k (para la rodilla)
        inercia = []
        k_range = range(1, len(self.centros_np) + 1)
        for k in k_range:
            temp = KMeans(n_clusters=k, init='k-means++', random_state=0)
            temp.fit(self.centros_np)
            inercia.append(temp.inertia_)

        # 3) Buscar rodilla si kneed está instalado
        try:
            from kneed import KneeLocator
            kneedle = KneeLocator(
                list(k_range),
                inercia,
                curve="convex",
                direction="decreasing"
            )
            k_optimo = kneedle.knee
            if k_optimo is None:
                k_optimo = self.k_means_clusters
        except ImportError:
            k_optimo = self.k_means_clusters

        # 4) Volver a entrenar con k_optimo
        kmeans_final = KMeans(n_clusters=k_optimo, init='k-means++', random_state=0)
        kmeans_final.fit(self.centros_np)
        self.etiquetas = kmeans_final.labels_
        self.centros_kmeans = kmeans_final.cluster_centers_


    def create_dataframe(self):
        """
        Construye un DataFrame con columnas:
          ['id', 'type', 'x', 'y', 'z', 'Cluster', 'NumAtomos']
        a partir de self.centros_np, self.etiquetas y self.lista_num_atomos.
        """
        self.df_clusters = pd.DataFrame({
            'id': np.arange(len(self.centros_np)),
            'x': self.centros_np[:, 0],
            'y': self.centros_np[:, 1],
            'z': self.centros_np[:, 2],
            'Cluster': self.etiquetas,
            'NumAtomos': self.lista_num_atomos
        })
        # Siempre el mismo 'type' = 1 para todos
        self.df_clusters['type'] = 1
        # Reordenar columnas
        self.df_clusters = self.df_clusters[['id', 'type', 'x', 'y', 'z', 'Cluster', 'NumAtomos']]


    def update_dump_header(self, key_areas_dump_path: str = "outputs.dump/key_areas.dump"):
        """
        Lee las primeras líneas de key_areas.dump hasta 'ITEM: ATOMS' y luego
        modifica la línea de número de átomos (si está) y la cabecera de átomos
        para que coincida con el formato:
          ITEM: ATOMS id type x y z etiqueta
        e inserta el nuevo número de átomos (len(self.df_clusters)).
        Finalmente, almacena este encabezado en self.new_header.
        """
        header_lines = []
        try:
            with open(key_areas_dump_path, "r", encoding="utf-8") as f:
                for line in f:
                    header_lines.append(line.rstrip("\n"))
                    if line.strip().startswith("ITEM: ATOMS"):
                        break
        except FileNotFoundError:
            raise FileNotFoundError(f"No se encontró el dump de áreas clave: {key_areas_dump_path}")

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

        # Reemplazar la línea 'ITEM: ATOMS ...' por 'ITEM: ATOMS id type x y z etiqueta'
        for idx, line in enumerate(new_header):
            if line.startswith("ITEM: ATOMS"):
                new_header[idx] = "ITEM: ATOMS id type x y z etiqueta"
                break

        self.new_header = new_header


    def export_dump_file(self, output_dump_path: str = "outputs.dump/city_population.dump"):
        """
        Escribe en disco un nuevo dump llamado output_dump_path:
        - Primero todas las líneas de self.new_header, luego los datos de self.df_clusters
          en formato LAMMPS dump (separado por espacio, sin encabezado ni índice).
        """
        os.makedirs(os.path.dirname(output_dump_path), exist_ok=True)
        with open(output_dump_path, "w", encoding="utf-8") as f:
            for line in self.new_header:
                f.write(line + "\n")
            # Exportar el DataFrame sin índice ni nombres de columna
            self.df_clusters.to_csv(f, sep=" ", index=False, header=False)


    def calculate_silhouette(self):
        """
        Calcula el valor de silhouette para cada centro de cluster en self.centros_np
        usando las etiquetas self.etiquetas y agrega la columna 'Silhouette' a self.df_clusters.
        Además guarda el promedio por cluster en self.silhouette_by_cluster.
        """
        from sklearn.metrics import silhouette_samples

        silhouette_vals = silhouette_samples(self.centros_np, self.etiquetas)
        self.df_clusters['Silhouette'] = silhouette_vals
        self.silhouette_by_cluster = self.df_clusters.groupby('Cluster')['Silhouette'].mean()


    def export_mapping(self, output_folder: str = "outputs.csv"):
        """
        Genera un CSV con el mapeo entre nombre de dump y cluster asignado,
        usando columnas ['Archivo', 'Cluster'] y lo guarda en output_folder.
        """
        os.makedirs(output_folder, exist_ok=True)
        mapping_df = pd.DataFrame({
            'Archivo': self.clusters_final,
            'Cluster': self.df_clusters['Cluster']
        })
        mapping_df.to_csv(
            os.path.join(output_folder, "mapeo_archivos_cluster.csv"),
            index=False
        )


    def export_figures(self, output_folder: str = "outputs.csv"):
        """
        Genera y guarda tres tipos de figuras:
         1) Gráfico 3D de densidad (heatmap) usando las coordenadas x,y,z.
         2) Tres mapas de contorno 2D (x-y, y-z, z-x).
         3) Gráfico de barras de población total por cluster.
         4) Mapa de calor (heatmap) de población por cluster.

        Todas las figuras se guardan en output_folder, usando el base name de 'defect'
        como prefijo en los nombres de archivo.
        """
        os.makedirs(output_folder, exist_ok=True)

        # Determinar nombre base a partir de self.defect
        defect_name = os.path.basename(self.defect)

        # 1) Gráfico 3D de densidad
        import scipy.stats as stats

        x = self.df_clusters['x']
        y = self.df_clusters['y']
        z = self.df_clusters['z']
        positions = np.vstack([x, y, z])
        density = stats.gaussian_kde(positions)(positions)

        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        sc = ax_3d.scatter(
            x, y, z,
            c=density,
            cmap='inferno',
            s=50,
            edgecolor='black'
        )
        fig_3d.colorbar(sc, ax=ax_3d, label='Densidad')
        ax_3d.set_title("Mapa 3D de densidad de centros")
        ax_3d.set_xlabel("x [Å]")
        ax_3d.set_ylabel("y [Å]")
        ax_3d.set_zlabel("z [Å]")

        output_file_3d = os.path.join(output_folder, f"{defect_name}_3D_heatmap.png")
        fig_3d.savefig(output_file_3d)
        plt.close(fig_3d)

        # 2) Mapas de contorno 2D (x-y, y-z, z-x)
        bins = 50
        x_arr = x.values
        y_arr = y.values
        z_arr = z.values

        xy_hist, xedges, yedges = np.histogram2d(x_arr, y_arr, bins=bins)
        extent_xy = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        yz_hist, yedges2, zedges = np.histogram2d(y_arr, z_arr, bins=bins)
        extent_yz = [yedges2[0], yedges2[-1], zedges[0], zedges[-1]]
        zx_hist, zedges2, xedges2 = np.histogram2d(z_arr, x_arr, bins=bins)
        extent_zx = [zedges2[0], zedges2[-1], xedges2[0], xedges2[-1]]

        fig_contour, axs = plt.subplots(1, 3, figsize=(18, 5))

        cset_xy = axs[0].contourf(xy_hist.T, levels=20, extent=extent_xy, cmap='viridis')
        axs[0].set_xlabel("x")
        axs[0].set_ylabel("y")
        axs[0].grid(True)
        fig_contour.colorbar(cset_xy, ax=axs[0], shrink=0.8)

        cset_yz = axs[1].contourf(yz_hist.T, levels=20, extent=extent_yz, cmap='viridis')
        axs[1].set_xlabel("y")
        axs[1].set_ylabel("z")
        axs[1].grid(True)
        fig_contour.colorbar(cset_yz, ax=axs[1], shrink=0.8)

        cset_zx = axs[2].contourf(zx_hist.T, levels=20, extent=extent_zx, cmap='viridis')
        axs[2].set_xlabel("z")
        axs[2].set_ylabel("x")
        axs[2].grid(True)
        fig_contour.colorbar(cset_zx, ax=axs[2], shrink=0.8)

        plt.tight_layout()
        output_file_contours = os.path.join(output_folder, f"{defect_name}_contour_maps.png")
        fig_contour.savefig(output_file_contours)
        plt.close(fig_contour)

        # 3) Gráfico de barras de población total por cluster
        poblacion_por_cluster = self.df_clusters.groupby('Cluster')['NumAtomos'].sum()
        poblacion_por_cluster_sorted = poblacion_por_cluster.sort_values(ascending=False)

        plt.figure(figsize=(12, 6))
        poblacion_por_cluster_sorted.plot(kind='bar')
        plt.xlabel("Cluster ID")
        plt.ylabel("Número total de átomos")
        plt.title("Población total por cluster")
        plt.grid(True)
        plt.tight_layout()
        bar_plot_file = os.path.join(output_folder, f"{defect_name}_pop_cluster_bar.png")
        plt.savefig(bar_plot_file)
        plt.close()

        # 4) Mapa de calor de población por cluster
        df_pop = poblacion_por_cluster_sorted.reset_index()
        df_pop.columns = ['Cluster', 'NumAtomos']

        plt.figure(figsize=(len(df_pop) * 0.5 + 2, 2))
        # Importamos seaborn solo si está disponible
        try:
            import seaborn as sns
            sns.heatmap(df_pop[['NumAtomos']].T, annot=True, fmt="d", cmap='viridis', cbar=True)
            plt.xlabel("Cluster")
            plt.ylabel("Número total de átomos")
            plt.title("Heatmap de población por cluster")
            plt.tight_layout()
            heatmap_file = os.path.join(output_folder, f"{defect_name}_pop_cluster_heatmap.png")
            plt.savefig(heatmap_file)
            plt.close()
        except ImportError:
            # Si seaborn no está instalado, simplemente no generamos el heatmap
            pass
