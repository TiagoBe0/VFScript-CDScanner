import os
import json
from ovito.io import import_file, export_file
from ovito.modifiers import DeleteSelectedModifier, ExpressionSelectionModifier
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
