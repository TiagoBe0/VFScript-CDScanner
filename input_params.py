# CONFIG: Lista de diccionarios con la configuración para el procesamiento y entrenamiento.
CONFIG = [{
    'training_step':False,
    # Ruta al archivo de entrada para el cálculo de relajación de la estructura.
    'relax' : 'inputs.dump/fe/fe_ful.data',
    'defect' : ['inputs.dump/dump-finalCool.160000'],# Ruta al archivo de entrada que describe el defecto o vacancia (por ejemplo, void_7).
    #'defect' : ['inputs.dump/dump-finalCool.160000','inputs.dump/Trackfcc/dump-finalCoolTrack.160000','inputs.dump/Trackfcc/data.sistema_track_final_cooled','inputs.dump/fcc+void/dump.ensayo.0.config','inputs.dump/fcc+void/dump.ensayo.50000.config','dump.ensayo.100000.config'],
    # Parámetro de radio para cálculos de área de influencia u otros procesos geométricos.
    #'defect' : ['inputs.dump/fe/dump.stp.0','inputs.dump/fe/dump.stp.5000','inputs.dump/fe/dump.stp.10000','inputs.dump/fe/dump.stp.0','inputs.dump/fe/dump.stp.15000','inputs.dump/fe/dump.stp.20000','inputs.dump/fe/dump.stp.25000','inputs.dump/fe/dump.stp.30000','inputs.dump/fe/dump.stp.35000','inputs.dump/fe/dump.stp.40000','inputs.dump/fe/dump.stp.45000','inputs.dump/fe/dump.stp.50000','inputs.dump/fe/dump.stp.55000','inputs.dump/fe/dump.stp.60000','inputs.dump/fe/dump.stp.65000','inputs.dump/fe/dump.stp.70000','inputs.dump/fe/dump.stp.75000','inputs.dump/fe/dump.stp.80000','inputs.dump/fe/dump.stp.85000','inputs.dump/fe/dump.stp.90000','inputs.dump/fe/dump.stp.95000','inputs.dump/fe/dump.stp.100000'],
    
    'radius' : 2,
    
    # Nivel de suavizado aplicado a los datos durante el preprocesamiento.
    'smoothing level' :0, 
    
    # Nivel de suavizado específico para los datos de entrenamiento.
    'smoothing_level_training' :0, 
    
    # Radio de corte: define el límite a partir del cual se ignoran interacciones o se delimita el área.
    'cutoff radius' : 6, 
    
    # Radio utilizado durante el entrenamiento para la selección o generación de datos.
    'radius_training': 3,
    
    # Indicador para activar o no métodos alternativos (por ejemplo, utilizar otro algoritmo de predicción).
    'other method': True,
    #Activar si se quiere acomular el entrenamiento cada iteracion
    'save_training':True,
    
    # Lista de parámetros de "strees" , si la muestra defectuosa sufrio una compresion del 20% en el eje x seria [0.8,1,1]
    'strees': [1, 1,1],
    
    # Tolerancia en la agrupación (clustering): define la distancia máxima para considerar puntos en el mismo grupo.
    'cluster tolerance':1.2,
    
    # Factor de división del cluster: puede ajustarse para definir cuántas particiones se desean (por ejemplo, 1, 2 o 3).
    'divisions_of_cluster': 6,  # Valores sugeridos: 1, 2 o 3.

    'CDScanner':True, #graficos de defectos

    'Histograma': True,
    
    # Si es True, se generarán datos sintéticos (generic data) para ampliar el conjunto de entrenamiento.
    'generic_data':     True,
    
    # Indicador para activar o no funcionalidades de chat o interacciones en modo conversacional.
    'chat': False,

    'iteraciones_clusterig':4
}]

# PREDICTOR_COLUMNS: Lista de columnas que se utilizarán para entrenar el modelo predictivo.
PREDICTOR_COLUMNS = ['surface_area','filled_volume','cluster_size','mean_distance']
#PREDICTOR_COLUMNS = ['sm_mesh_training', 'filled_volume', 'vecinos']