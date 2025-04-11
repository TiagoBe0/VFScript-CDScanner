import numpy as np
import json

class CrystalStructureGenerator:
    def __init__(self, config_file="input_params.json"):
        self.config_file = config_file
        self.structure_type = None
        self.lattice_parameter = None
        self.load_config()
    
    def load_config(self):
        """Carga la configuración desde el archivo JSON"""
        with open(self.config_file, "r") as f:
            json_data = json.load(f)
            config = json_data["CONFIG"][0]
            relax = config['generate_relax']
            self.structure_type = relax[0]
            self.lattice_parameter = float(relax[1])
    
    def generate_fcc(self, repetitions=(1,1,1)):
        """Genera estructura cristalina FCC"""
        # Posiciones atómicas base para FCC
        base_positions = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0],
            [0.5, 0, 0.5],
            [0, 0.5, 0.5],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1],
            [0.5, 0.5, 1],
            [0.5, 1, 0.5],
            [1, 0.5, 0.5]
        ]) * self.lattice_parameter
        
        return self._replicate_structure(base_positions, repetitions)
    
    def generate_bcc(self, repetitions=(1,1,1)):
        """Genera estructura cristalina BCC"""
        # Posiciones atómicas base para BCC
        base_positions = np.array([
            [0, 0, 0],
            [0.5, 0.5, 0.5],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]
        ]) * self.lattice_parameter
        
        return self._replicate_structure(base_positions, repetitions)
    
    def _replicate_structure(self, base_positions, repetitions):
        """Replica la estructura base según las repeticiones especificadas"""
        coordinates = []
        for i in range(repetitions[0]):
            for j in range(repetitions[1]):
                for k in range(repetitions[2]):
                    displacement = np.array([i, j, k]) * self.lattice_parameter
                    for atom in base_positions:
                        coordinates.append(atom + displacement)
        
        coordinates = np.array(coordinates)
        coordinates = np.unique(np.round(coordinates, decimals=6), axis=0)
        
        box_limits = (
            0.0, self.lattice_parameter * repetitions[0],
            0.0, self.lattice_parameter * repetitions[1],
            0.0, self.lattice_parameter * repetitions[2]
        )
        
        return coordinates, box_limits
    
    def save_to_dump(self, coordinates, box_limits, output_file):
        """Guarda la estructura en formato LAMMPS dump"""
        with open(output_file, "w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write("0\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{len(coordinates)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{box_limits[0]} {box_limits[1]}\n")
            f.write(f"{box_limits[2]} {box_limits[3]}\n")
            f.write(f"{box_limits[4]} {box_limits[5]}\n")
            f.write("ITEM: ATOMS id type x y z\n")
            for i, coord in enumerate(coordinates, start=1):
                f.write(f"{i} 1 {coord[0]} {coord[1]} {coord[2]}\n")
    
    def generate(self, repetitions=(10,10,10)):
        """Genera la estructura según la configuración cargada"""
        if self.structure_type == "fcc":
            coordinates, box_limits = self.generate_fcc(repetitions)
            output_file = "estructura_fcc_multicelda.dump"
        elif self.structure_type == "bcc":
            coordinates, box_limits = self.generate_bcc(repetitions)
            output_file = "estructura_bcc_multicelda.dump"
        else:
            raise ValueError(f"Tipo de estructura no soportado: {self.structure_type}")
        
        self.save_to_dump(coordinates, box_limits, output_file)
        
        # Mostrar información de la estructura generada
        print(f"\nEstructura {self.structure_type.upper()} generada exitosamente")
        print(f"Archivo de salida: {output_file}")
        print(f"Parámetro de red: {self.lattice_parameter}")
        print(f"Celdas unitarias: {repetitions[0]}x{repetitions[1]}x{repetitions[2]}")
        print(f"Total de átomos: {len(coordinates)}")
        print(f"Límites de la caja:")
        print(f"  X: {box_limits[0]} - {box_limits[1]}")
        print(f"  Y: {box_limits[2]} - {box_limits[3]}")
        print(f"  Z: {box_limits[4]} - {box_limits[5]}")


# Ejemplo de uso
if __name__ == "__main__":
    generator = CrystalStructureGenerator()
    generator.generate(repetitions=(10, 10, 10))