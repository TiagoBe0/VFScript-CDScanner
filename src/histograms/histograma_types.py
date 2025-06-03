import numpy as np
import matplotlib.pyplot as plt

class HistogramaTypes:
    def __init__(self, resultados):
        self.resultados = resultados

    def obtener_tipos_unicos(self):
        tipos_unicos = set()
        for resultado in self.resultados:
            tipos_unicos.update(resultado['Conteo_Tipos'].keys())
        return sorted(tipos_unicos, key=lambda x: int(x))

    def crear_matriz_frecuencias(self, clusters_unicos, tipos_unicos):
        matriz_frecuencias = []
        for cluster in clusters_unicos:
            frecuencias_cluster = {tipo: 0 for tipo in tipos_unicos}
            for resultado in self.resultados:
                if resultado['Cluster'] == cluster:
                    for tipo, frecuencia in resultado['Conteo_Tipos'].items():
                        frecuencias_cluster[tipo] += frecuencia
            fila = [cluster] + [frecuencias_cluster[tipo] for tipo in tipos_unicos]
            matriz_frecuencias.append(fila)
        return matriz_frecuencias

    def graficar(self, matriz_frecuencias, tipos_unicos, nombre_archivo):
        suma_frecuencias = [sum(fila[1:]) for fila in matriz_frecuencias]
        indices_ordenados = np.argsort(suma_frecuencias)[::-1]
        matriz_frecuencias = [matriz_frecuencias[i] for i in indices_ordenados]
        clusters_ordenados = [fila[0] for fila in matriz_frecuencias]
        x_positions = np.arange(len(clusters_ordenados))
        frecuencias = np.array([fila[1:] for fila in matriz_frecuencias])
        fig, ax = plt.subplots(figsize=(10, 6))
        bottom = np.zeros(len(clusters_ordenados))
        for i, tipo in enumerate(tipos_unicos):
            ax.bar(x_positions, frecuencias[:, i], bottom=bottom, label=f'Tipo {tipo}')
            bottom += frecuencias[:, i]
        for j, pos in enumerate(x_positions):
            total = np.sum(frecuencias[j])
            if total > 0:
                acumulado = 0
                for i, valor in enumerate(frecuencias[j]):
                    if valor > 0:
                        porcentaje = (valor / total) * 100
                        y_pos = acumulado + valor / 2
                        ax.text(pos, y_pos, f'{porcentaje:.1f}%', ha='center', va='center', color='black', fontsize=12, fontweight='bold')
                    acumulado += valor
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Frecuencia de átomos')
        ax.set_xticks(x_positions)
        ax.set_xticklabels(clusters_ordenados)
        ax.legend(title='Tipos de átomos', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.grid(True)
        nombre_sin_ext = os.path.splitext(os.path.basename(nombre_archivo))[0]
        plt.savefig(f'{nombre_sin_ext}.png')
        plt.show()

