import json
import numpy as np



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