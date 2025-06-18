import os
import csv
import numpy as np

def load_norms_csv(csv_path):
    """
    Lee un CSV donde cada fila comienza con un nombre de archivo
    seguido de una lista de normas ordenadas. Retorna un diccionario:
      { file_name: np.array([norm_1, norm_2, ...]) }
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"No se encontró el archivo CSV:\n  {csv_path}")

    data = {}
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            file_name = row[0]
            # Convertir el resto de columnas en float
            try:
                norms = np.array([float(v) for v in row[1:]])
            except ValueError:
                # Si hay algún valor inválido, saltar esa fila
                continue
            data[file_name] = norms
    return data

def find_best_matches(defect_data, train_data):
    """
    Para cada entrada en defect_data (un dict {file: norms_array}), 
    busca en train_data (otro dict {file: norms_array}) el mejor 'match'
    según distancia euclídea entre los vectores de normas.
    Retorna un diccionario:
      { defect_file: (best_train_file, min_distance) }
    """
    matches = {}
    for d_file, d_norms in defect_data.items():
        best_file = None
        best_dist = None

        for t_file, t_norms in train_data.items():
            # Si los arrays tienen distinta longitud, recortamos al mínimo
            min_len = min(len(d_norms), len(t_norms))
            if min_len == 0:
                continue

            # Tomamos las primeras min_len normas de cada uno
            a = d_norms[:min_len]
            b = t_norms[:min_len]
            dist = np.linalg.norm(a - b)

            if (best_dist is None) or (dist < best_dist):
                best_dist = dist
                best_file = t_file

        if best_file is not None:
            matches[d_file] = (best_file, best_dist)
        else:
            matches[d_file] = (None, None)
    return matches

def export_matches_to_csv(matches_dict, output_csv):
    """
    Dado un dict { defect_file: (train_file, distance) }, lo vuelca
    a un CSV con columnas: defect_file, best_train_file, distance
    """
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["defect_file", "best_train_file", "distance"])
        for d_file, (t_file, dist) in matches_dict.items():
            if t_file is None:
                writer.writerow([d_file, "", ""])
            else:
                writer.writerow([d_file, t_file, f"{dist:.6f}"])
    print(f"Se exportaron los mejores matches en: {output_csv}")

def find_global_winner(matches_dict):
    """
    Dado un dict { defect_file: (train_file, distance) }, 
    devuelve el par (defect_file, train_file, distance) con la distancia mínima.
    """
    winner_defect = None
    winner_train = None
    winner_dist = None

    for d_file, (t_file, dist) in matches_dict.items():
        if t_file is None or dist is None:
            continue
        if (winner_dist is None) or (dist < winner_dist):
            winner_dist = dist
            winner_defect = d_file
            winner_train = t_file

    if winner_defect is None:
        return None  # No se encontró ningún match válido
    return (winner_defect, winner_train, winner_dist)

if __name__ == "__main__":
    # Rutas a los CSV generados previamente
    defect_csv = "outputs/csv/finger_data.csv"
    train_csv  = "outputs/csv/finger_defect_data.csv"

    # 1) Cargar los datos de normas
    defect_norms = load_norms_csv(defect_csv)
    train_norms  = load_norms_csv(train_csv)

    # 2) Encontrar el mejor match para cada defecto
    best_matches = find_best_matches(defect_norms, train_norms)

    # 3) Exportar resultados a CSV
    output_matches_csv = "outputs/csv/best_finger_matches.csv"
    os.makedirs(os.path.dirname(output_matches_csv), exist_ok=True)
    export_matches_to_csv(best_matches, output_matches_csv)

    # 4) Elegir un ganador global
    winner = find_global_winner(best_matches)
    if winner is None:
        print("No se encontró ningún match válido para declarar un ganador global.")
    else:
        defect_winner, train_winner, dist_winner = winner
        print("\n===== Ganador global =====")
        print(f"Defecto:  {defect_winner}")
        print(f"Entrenamiento: {train_winner}")
        print(f"Distancia: {dist_winner:.6f}")
        print("==========================")
