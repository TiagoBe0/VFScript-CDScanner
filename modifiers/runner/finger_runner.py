#!/usr/bin/env python3
"""
finger_runner_oop.py

Clase orientada a objetos para comparar dos CSVs:
- `outputs/csv/finger_defect_data.csv`
- `outputs/csv/finger_data.csv`

Calcula la fila más parecida por distancia euclídea sobre columnas numéricas,
excluyendo la columna identificadora, y exporta un CSV con los ganadores.
En la columna `winner_file` se extrae únicamente el número presente en el nombre del archivo ganador.
"""
import re
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
from pandas.api.types import is_numeric_dtype

class WinnerFinger:
    """
    Orquesta la comparación de dos CSVs y guarda un CSV con los resultados.
    """
    def __init__(self,
                 defect_csv: Path,
                 normal_csv: Path,
                 output_csv: Path,
                 id_col: str = 'file_name'):
        self.defect_csv = defect_csv
        self.normal_csv = normal_csv
        self.output_csv = output_csv
        self.id_col = id_col
        self.defect_df = None
        self.normal_df = None
        self.numeric_cols = []
        self.results_df = None

    def load_data(self):
        """Carga los DataFrames desde los archivos CSV."""
        if not self.defect_csv.is_file():
            raise FileNotFoundError(f"No se encontró el CSV de defectos en: {self.defect_csv}")
        if not self.normal_csv.is_file():
            raise FileNotFoundError(f"No se encontró el CSV normal en: {self.normal_csv}")
        self.defect_df = pd.read_csv(self.defect_csv)
        self.normal_df = pd.read_csv(self.normal_csv)

    def validate_id_column(self):
        """Verifica que la columna identificadora exista en ambos DataFrames."""
        for df, name in [(self.defect_df, 'defect_df'), (self.normal_df, 'normal_df')]:
            if self.id_col not in df.columns:
                raise KeyError(
                    f"La columna identificadora '{self.id_col}' no está en {name}. "
                    f"Columnas disponibles: {list(df.columns)}"
                )

    def select_numeric_columns(self):
        """Determina las columnas numéricas comunes para comparar."""
        cols = []
        for col in self.defect_df.columns:
            if col == self.id_col:
                continue
            if col in self.normal_df.columns and \
               is_numeric_dtype(self.defect_df[col]) and \
               is_numeric_dtype(self.normal_df[col]):
                cols.append(col)
        if not cols:
            raise ValueError(
                "No se encontraron columnas numéricas para comparar. "
                f"Defect num: {[c for c in self.defect_df.columns if is_numeric_dtype(self.defect_df[c])]}; "
                f"Normal num: {[c for c in self.normal_df.columns if is_numeric_dtype(self.normal_df[c])]}."
            )
        self.numeric_cols = cols

    def extract_number(self, filename: str) -> int:
        """Extrae el primer número entero del nombre de archivo."""
        m = re.search(r"(\d+)", filename)
        return int(m.group(1)) if m else None

    def compute_winners(self):
        """Para cada fila defectuosa, encuentra la fila normal con mínima distancia euclídea."""
        norm_vals = self.normal_df[self.numeric_cols].to_numpy(dtype=float)
        norm_ids  = self.normal_df[self.id_col].tolist()
        results = []

        for _, row in self.defect_df.iterrows():
            defect_id   = row[self.id_col]
            defect_vals = row[self.numeric_cols].to_numpy(dtype=float)
            diffs = norm_vals - defect_vals
            dists = np.linalg.norm(diffs, axis=1)
            idx = int(np.argmin(dists))
            winner_filename = norm_ids[idx]
            winner_number = self.extract_number(winner_filename)
            results.append({
                'defect_file': defect_id,
                'fingerprint': winner_number,
                'distance':    float(dists[idx])
            })
        self.results_df = pd.DataFrame(results)

    def save_results(self):
        """Guarda el DataFrame de resultados a CSV."""
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        self.results_df.to_csv(self.output_csv, index=False)

    def run(self):
        """Ejecuta todos los pasos del pipeline: cargar, validar, seleccionar, calcular y guardar."""
        self.load_data()
        self.validate_id_column()
        self.select_numeric_columns()
        self.compute_winners()
        self.save_results()
        print(f"Guardados {len(self.results_df)} ganadores en {self.output_csv}")



def main():
    parser = argparse.ArgumentParser(
        description="Compara filas defectuosas vs normales y exporta ganadores."
    )
    parser.add_argument('--id-col', default='file_name',
                        help="Nombre de la columna identificadora (por defecto 'file_name')")
    parser.add_argument('--out', default='outputs/csv/resultados_ganadores.csv',
                        help="Ruta de salida para el CSV de ganadores")
    args = parser.parse_args()

    base = Path(__file__).parents[2] / 'outputs' / 'csv'
    defect_path = base / 'finger_defect_data.csv'
    normal_path = base / 'finger_data.csv'
    output_path = Path(args.out)

    finder = WinnerFinger(defect_path, normal_path, output_path, id_col=args.id_col)
    finder.run()
    # 1. Rutas a tus archivos
    defect_csv   = Path('outputs/csv/defect_data.csv')
    finger_csv   = Path('outputs/csv/finger_winner_data.csv')
    output_csv   = Path('outputs/csv/defect_data_enriched.csv')  # salida

    # 2. Carga de datos
    defect_df = pd.read_csv(defect_csv)
    finger_df = pd.read_csv(finger_csv)

    # 3. Extraer sólo las columnas clave de finger_df
    #    Asumimos que en defect_df la columna 'archivo' tiene rutas como ".../key_area_1.dump"
    #    y en finger_df la columna 'defect_file' sólo el nombre "key_area_1.dump".
    finger_df = finger_df[['defect_file', 'fingerprint']]

    # 4. Crear en defect_df una columna con sólo el nombre de archivo
    defect_df['file_name'] = defect_df['archivo'].apply(lambda p: Path(p).name)

    # 5. Hacer merge para añadir 'fingerprint'
    enriched = defect_df.merge(
        finger_df,
        how='left',
        left_on='file_name',
        right_on='defect_file'
    )

    # 6. Limpiar columnas auxiliares y reordenar si lo deseas
    enriched = enriched.drop(columns=['file_name', 'defect_file'])

    # 7. Guardar resultado
    enriched.to_csv(output_csv, index=False)
    print(f'CSV enriquecido guardado en: {output_csv}')

if __name__ == '__main__':
    main()
