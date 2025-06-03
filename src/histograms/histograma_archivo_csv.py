import csv
import csv

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
