import os
import csv
import numpy as np

class DumpProcessor:
    """
    Se encarga de leer un archivo .dump de LAMMPS, desplazar las coordenadas
    al centro de masa y devolver las normas de las coordenadas desplazadas.
    """
    def __init__(self, dump_path: str):
        self.dump_path = dump_path
        self.coords_originales = None      # Coordenadas originales (numpy array Nx3)
        self.center_of_mass = None         # Centro de masa (tuple de 3 elementos)
        self.coords_trasladadas = None     # Coordenadas trasladadas al origen (numpy Nx3)
        self.norms = None                 # Normas de coords_trasladadas (numpy array de tamaño N)

    def read_and_translate(self):
        """
        Lee el archivo .dump y traslada las coordenadas de modo que el
        centro de masa quede en el origen. Guarda en los atributos:
          - self.coords_originales
          - self.center_of_mass
          - self.coords_trasladadas
        """
        if not os.path.isfile(self.dump_path):
            raise FileNotFoundError(f"No se encontró el archivo: {self.dump_path}")

        coords = []
        with open(self.dump_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        # Encontrar el inicio de la sección de átomos
        start_index = None
        for i, line in enumerate(lines):
            if line.strip().startswith("ITEM: ATOMS"):
                start_index = i + 1
                break

        if start_index is None:
            raise ValueError(f"No se encontró 'ITEM: ATOMS' en {self.dump_path}")

        # Leer coordenadas (x, y, z) línea por línea hasta la siguiente sección
        for line in lines[start_index:]:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "ITEM:":
                break  # fin de la sección de átomos
            if len(parts) < 5:
                continue
            try:
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                coords.append((x, y, z))
            except ValueError:
                # Si la línea no contiene números válidos, ignorarla
                continue

        if not coords:
            raise ValueError(f"No se hallaron coordenadas válidas tras 'ITEM: ATOMS' en {self.dump_path}")

        # Convertir a numpy array y calcular centro de masa
        self.coords_originales = np.array(coords)
        com = tuple(self.coords_originales.mean(axis=0))
        self.center_of_mass = com

        # Trasladar coordenadas restando el centro de masa
        self.coords_trasladadas = self.coords_originales - np.array(com)

    def compute_norms(self):
        """
        Calcula la norma de cada vector de coordenadas trasladadas.
        Debe llamarse después de read_and_translate().
        Guarda el resultado ordenado en self.norms (numpy array de tamaño N).
        """
        if self.coords_trasladadas is None:
            raise RuntimeError("Debes llamar primero a read_and_translate() antes de compute_norms().")

        # Calcular normas de cada vector fila (distancia al origen)
        self.norms = np.linalg.norm(self.coords_trasladadas, axis=1)
        # Ordenar las normas de menor a mayor
        self.norms = np.sort(self.norms)


class StatisticsCalculator:
    """
    Calcula un conjunto de estadísticas (min, max, mean, std, skewness, kurtosis,
    percentiles, IQR, histograma normalizado) sobre un array 1D de valores.
    """
    @staticmethod
    def compute_statistics(arr: np.ndarray) -> dict:
        stats = {}
        N = len(arr)
        stats['N'] = N

        if N == 0:
            # Si no hay elementos, rellenar con NaN y bins en cero
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

        # Skewness: E[((x-μ)/σ)^3]
        skew_val = float(np.mean(((arr - mean_val) / std_val) ** 3)) if std_val > 0 else 0.0
        # Kurtosis (exceso): E[((x-μ)/σ)^4] - 3
        kurt_val = float(np.mean(((arr - mean_val) / std_val) ** 4) - 3) if std_val > 0 else 0.0

        Q1 = float(np.percentile(arr, 25))
        med = float(np.percentile(arr, 50))
        Q3 = float(np.percentile(arr, 75))
        IQR = Q3 - Q1

        # Histograma de 10 bins entre min y max (densidad relativa)
        hist_counts, _ = np.histogram(arr, bins=10, range=(min_val, max_val))
        hist_norm = hist_counts / N  # normalizado a proporciones

        stats.update({
            'min': min_val,
            'max': max_val,
            'mean': mean_val,
            'std': std_val,
            'skewness': skew_val,
            'kurtosis': kurt_val,
            'Q1': Q1,
            'median': med,
            'Q3': Q3,
            'IQR': IQR
        })
        for i, h in enumerate(hist_norm, start=1):
            stats[f'hist_bin_{i}'] = float(h)

        return stats


class FeatureExporter:
    """
    Recorre una lista de archivos .dump, utiliza DumpProcessor para
    extraer normas y StatisticsCalculator para obtener estadísticas,
    y finalmente escribe un CSV con todas las características.
    """
    def __init__(self, dump_paths: list[str], output_csv: str):
        self.dump_paths = dump_paths
        self.output_csv = output_csv

    def export(self):
        # Definir encabezados del CSV
        header = [
            "file_name", "N", "min", "max", "mean", "std",
            "skewness", "kurtosis", "Q1", "median", "Q3", "IQR"
        ] + [f"hist_bin_{i}" for i in range(1, 11)]

        # Asegurar que el directorio de salida existe
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)

        with open(self.output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(header)

            for dump_path in self.dump_paths:
                if not os.path.isfile(dump_path):
                    print(f"Advertencia: no se encontró {dump_path}, se salta este archivo.")
                    continue

                # Procesar el archivo dump
                processor = DumpProcessor(dump_path)
                try:
                    processor.read_and_translate()
                    processor.compute_norms()
                except Exception as e:
                    print(f"Error procesando {dump_path}: {e}")
                    continue

                # Obtener normas y calcular estadísticas
                norms = processor.norms
                stats = StatisticsCalculator.compute_statistics(norms)

                # Construir la fila de salida con todas las características
                file_name = os.path.basename(dump_path)
                row = [
                    file_name,
                    stats['N'],
                    stats['min'],
                    stats['max'],
                    stats['mean'],
                    stats['std'],
                    stats['skewness'],
                    stats['kurtosis'],
                    stats['Q1'],
                    stats['median'],
                    stats['Q3'],
                    stats['IQR']
                ]
                # Agregar los 10 bins del histograma
                for i in range(1, 11):
                    row.append(stats[f'hist_bin_{i}'])

                writer.writerow(row)

        print(f"Se generó el CSV con características en: {self.output_csv}")


# Ejemplo de uso
if __name__ == "__main__":
    # Lista de archivos .dump de entrenamiento (vacancias) a procesar
    dump_files = [
        "outputs/dump/vacancy_1_training.dump",
        "outputs/dump/vacancy_2_training.dump",
        "outputs/dump/vacancy_3_training.dump",
        "outputs/dump/vacancy_4_training.dump",
        "outputs/dump/vacancy_5_training.dump",
        "outputs/dump/vacancy_6_training.dump",
        "outputs/dump/vacancy_7_training.dump",
        "outputs/dump/vacancy_8_training.dump",
        "outputs/dump/vacancy_9_training.dump",
        "outputs/dump/vacancy_10_training.dump",
        "outputs/dump/vacancy_11_training.dump",
        "outputs/dump/vacancy_12_training.dump",
        "outputs/dump/vacancy_13_training.dump",
        "outputs/dump/vacancy_14_training.dump"
    ]
    output_csv_path = "outputs/csv/finger_data.csv"

    exporter = FeatureExporter(dump_files, output_csv_path)
    exporter.export()
