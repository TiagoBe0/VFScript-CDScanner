import os
import json
import csv
import numpy as np

def read_dump_and_translate_to_com(dump_path):
    """
    Lee un archivo .dump de LAMMPS y traslada sus coordenadas 
    para que el centro de masa quede en el origen.

    Retorna:
      - coords_originales: array numpy (N, 3) con coordenadas originales.
      - center_of_mass: tupla (com_x, com_y, com_z).
      - coords_trasladadas: array numpy (N, 3) con coordenadas trasladas.
    """
    coords_originales = []
    with open(dump_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # 1) Buscar la línea "ITEM: ATOMS"
    start_index = None
    for i, line in enumerate(lines):
        if line.strip().startswith("ITEM: ATOMS"):
            start_index = i + 1
            break

    if start_index is None:
        raise ValueError(f"No se encontró 'ITEM: ATOMS' en {dump_path}")

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
        raise ValueError(f"No se hallaron coordenadas válidas tras 'ITEM: ATOMS' en {dump_path}")

    coords_originales = np.array(coords_originales)  # (N, 3)

    # 3) Calcular centro de masa (promedio de cada eje)
    com = tuple(coords_originales.mean(axis=0))  # (com_x, com_y, com_z)

    # 4) Trasladar las coordenadas restando el COM
    coords_trasladadas = coords_originales - np.array(com)

    return coords_originales, com, coords_trasladadas

def compute_statistics(norms):
    """
    Dado un array 1D de normas, calcula:
      - N: número de elementos
      - min, max, mean, std, skewness, kurtosis
      - Q1 (percentil 25), median (percentil 50), Q3 (percentil 75), IQR
      - Histograma con 10 bins (normalizado)
    Retorna un dict con todos estos valores.
    """
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
        # hist bins as zeros
        for i in range(1, 11):
            stats[f'hist_bin_{i}'] = 0.0
        return stats

    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=0))
    # Skewness: E[((x-μ)/σ)^3]
    skew_val = float(np.mean(((arr - mean_val) / std_val)**3)) if std_val > 0 else 0.0
    # Kurtosis (exceso): E[((x-μ)/σ)^4] - 3
    kurt_val = float(np.mean(((arr - mean_val) / std_val)**4) - 3) if std_val > 0 else 0.0

    Q1 = float(np.percentile(arr, 25))
    med = float(np.percentile(arr, 50))
    Q3 = float(np.percentile(arr, 75))
    IQR = Q3 - Q1

    # Histograma 10 bins entre min y max
    hist_counts, _ = np.histogram(arr, bins=10, range=(min_val, max_val))
    hist_norm = hist_counts / N  # normalizado

    stats.update({
        'min': min_val, 'max': max_val, 'mean': mean_val, 'std': std_val,
        'skewness': skew_val, 'kurtosis': kurt_val,
        'Q1': Q1, 'median': med, 'Q3': Q3, 'IQR': IQR
    })
    for i, h in enumerate(hist_norm, start=1):
        stats[f'hist_bin_{i}'] = float(h)
    return stats

def export_features_to_csv_from_json(json_path, output_csv):
    """
    Lee la lista de archivos .dump desde 'clusters_final' en json_path,
    calcula estadísticas sobre las normas trasladadas al COM, y escribe
    un CSV de características con:
      [file_name, N, min, max, mean, std, skewness, kurtosis,
       Q1, median, Q3, IQR, hist_bin_1, ..., hist_bin_10]
    """
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"No se encontró el archivo JSON:\n  {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    dump_list = data.get("clusters_final", [])
    if not isinstance(dump_list, list) or len(dump_list) == 0:
        raise KeyError(f"El JSON debe contener una lista no vacía en 'clusters_final'. Encontrado: {dump_list}")

    # Definir encabezados
    header = [
        "file_name", "N", "min", "max", "mean", "std",
        "skewness", "kurtosis", "Q1", "median", "Q3", "IQR"
    ] + [f"hist_bin_{i}" for i in range(1, 11)]

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)

        for dump_path in dump_list:
            if not os.path.isfile(dump_path):
                print(f"Advertencia: no se encontró {dump_path}, se salta este archivo.")
                continue

            # Leer y trasladar al COM
            _, com, coords_shifted = read_dump_and_translate_to_com(dump_path)
            norms = np.linalg.norm(coords_shifted, axis=1)
            norms_sorted = np.sort(norms)

            # Calcular estadísticas
            stats = compute_statistics(norms_sorted)

            # Preparar la fila
            file_name = os.path.basename(dump_path)
            row = [file_name,
                   stats['N'], stats['min'], stats['max'], stats['mean'], stats['std'],
                   stats['skewness'], stats['kurtosis'],
                   stats['Q1'], stats['median'], stats['Q3'], stats['IQR']]
            # Agregar bins del histograma
            for i in range(1, 11):
                row.append(stats[f'hist_bin_{i}'])

            writer.writerow(row)

    print(f"Se generó el CSV con características en: {output_csv}")

# === Ejecución ===
if __name__ == "__main__":
    json_input = "outputs/json/key_archivos.json"
    output_csv_path = "outputs/csv/finger_defect_data.csv"
    export_features_to_csv_from_json(json_input, output_csv_path)
