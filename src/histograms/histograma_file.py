
from collections import Counter

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