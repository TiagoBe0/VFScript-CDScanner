#!/usr/bin/env python3
"""
feature_exporter.py

Clase orientada a objetos para leer dumps desde un JSON, trasladar
coordenadas al centro de masa, calcular estadísticas y exportar un CSV.
Mantiene la lógica y rutas exactas del código original.
"""

import os
import json
import csv
import numpy as np

class DumpProcessorFinger:
    """
    Se encarga de leer un .dump de LAMMPS, trasladar sus coordenadas
    al centro de masa y devolver las coordenadas trasladadas.
    """
    def __init__(self, dump_path: str):
        self.dump_path = dump_path

    def read_and_translate_to_com(self):
        coords_originales = []
        with open(self.dump_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # 1) Buscar la línea "ITEM: ATOMS"
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break

        if start_index is None:
            raise ValueError(f"No se encontró 'ITEM: ATOMS' en {self.dump_path}")

        # 2) Extraer x, y, z de cada línea hasta la siguiente sección
        for line in lines[start_index:]:
            parts = line.split()
            if parts[0] == "ITEM:":
                break
            if len(parts) < 5:
                continue
            try:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                coords_originales.append((x, y, z))
            except ValueError:
                continue

        if not coords_originales:
            raise ValueError(f"No se hallaron coordenadas válidas tras 'ITEM: ATOMS' en {self.dump_path}")

        coords_originales = np.array(coords_originales)
        # 3) Calcular centro de masa
        com = tuple(coords_originales.mean(axis=0))
        # 4) Trasladar las coordenadas restando el COM
        coords_trasladadas = coords_originales - np.array(com)
        return coords_originales, com, coords_trasladadas

class StatisticsCalculatorFinger:
    """
    Calcula estadísticas y histograma normalizado sobre un array 1D de normas.
    """
    @staticmethod
    def compute_statistics(norms: np.ndarray) -> dict:
        stats = {}
        arr = norms
        N = len(arr)
        stats['N'] = N

        if N == 0:
            stats.update({
                'min': np.nan, 'max': np.nan, 'mean': np.nan, 'std': np.nan,
                'skewness': np.nan, 'kurtosis': np.nan,
                'Q1': np.nan, 'median': np.nan, 'Q3': np.nan, 'IQR': np.nan
            })
            for i in range(1, 11):
                stats[f'hist_bin_{i}'] = 0.0
            return stats

        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=0))
        skew_val = float(np.mean(((arr - mean_val) / std_val)**3)) if std_val > 0 else 0.0
        kurt_val = float(np.mean(((arr - mean_val) / std_val)**4) - 3) if std_val > 0 else 0.0

        Q1 = float(np.percentile(arr, 25))
        med = float(np.percentile(arr, 50))
        Q3 = float(np.percentile(arr, 75))
        IQR = Q3 - Q1

        hist_counts, _ = np.histogram(arr, bins=10, range=(min_val, max_val))
        hist_norm = hist_counts / N

        stats.update({
            'min': min_val, 'max': max_val, 'mean': mean_val, 'std': std_val,
            'skewness': skew_val, 'kurtosis': kurt_val,
            'Q1': Q1, 'median': med, 'Q3': Q3, 'IQR': IQR
        })
        for i, h in enumerate(hist_norm, start=1):
            stats[f'hist_bin_{i}'] = float(h)

        return stats

class JSONFeatureExporterFinger:
    """
    Lee 'clusters_final' de un JSON, procesa cada dump con DumpProcessor
    y StatisticsCalculator, y exporta un CSV con todas las características.
    """
    def __init__(self, json_path: str, output_csv: str):
        self.json_path = json_path
        self.output_csv = output_csv

    def export(self):
        # 1) Cargar JSON
        if not os.path.isfile(self.json_path):
            raise FileNotFoundError(f"No se encontró el archivo JSON:\n  {self.json_path}")
        with open(self.json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        dump_list = data.get("clusters_final", [])
        print(f"DEBUG: dumps list loaded from JSON ({len(dump_list)} items):")
        for p in dump_list:
            print(" ", p)

        # 2) Normalizar rutas
        normalized = []
        for p in dump_list:
            if os.path.dirname(p):
                normalized.append(p)
            else:
                normalized.append(os.path.join("outputs", "dump", p))
        dump_list = normalized

        # 3) Encabezados CSV
        header = [
            "file_name", "N", "min", "max", "mean", "std",
            "skewness", "kurtosis", "Q1", "median", "Q3", "IQR"
        ] + [f"hist_bin_{i}" for i in range(1, 11)]

        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        rows_written = 0

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for dump_path in dump_list:
                if not os.path.isfile(dump_path):
                    print(f"  ❌ Advertencia: no se encontró {dump_path}, se omite.")
                    continue

                try:
                    _, com, coords_shifted = DumpProcessorFinger(dump_path).read_and_translate_to_com()
                except Exception as e:
                    print(f"  ❌ Error al leer/trasladar {dump_path}: {e}")
                    continue

                norms = np.linalg.norm(coords_shifted, axis=1)
                norms_sorted = np.sort(norms)
                stats = StatisticsCalculatorFinger.compute_statistics(norms_sorted)

                file_name = os.path.basename(dump_path)
                row = [
                    file_name,
                    stats['N'], stats['min'], stats['max'], stats['mean'], stats['std'],
                    stats['skewness'], stats['kurtosis'],
                    stats['Q1'], stats['median'], stats['Q3'], stats['IQR']
                ]
                for i in range(1, 11):
                    row.append(stats[f'hist_bin_{i}'])

                writer.writerow(row)
                rows_written += 1

        print(f"✅ Se generó el CSV en '{self.output_csv}' con {rows_written} filas.")

# === Ejecución ===
if __name__ == "__main__":
    json_input      = "outputs/json/key_archivos.json"
    output_csv_path = "outputs/csv/finger_defect_data.csv"
    exporter = JSONFeatureExporterFinger(json_input, output_csv_path)
    exporter.export()
