import numpy as np
from pathlib import Path
from typing import List, Tuple  
import numpy as np
from pathlib import Path

class CrystalStructureGenerator:
    """Genera la estructura relajada según el config y exporta un dump LAMMPS."""
    def __init__(self, config: dict, out_dir: Path):
        self.config = config
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)

        # 0) tipo de red: "fcc" o "bcc"
        self.structure_type = config['generate_relax'][0].lower()

        # 1) parámetro de red (lattice parameter)
        self.lattice = float(config['generate_relax'][1])

        # 2,3,4) réplicas en X, Y, Z
        #    Parsear tres enteros
        rx = int(config['generate_relax'][2])
        ry = int(config['generate_relax'][3])
        rz = int(config['generate_relax'][4])
        self.reps = (rx, ry, rz)

    def generate(self) -> Path:
        """
        Construye la réplica con las tres dimensiones self.reps.
        Guarda relax_structure.dump dentro de out_dir.
        """
        coords, dims = self._build_replica(self.reps)
        centered = coords - dims/2
        box = (
            -dims[0]/2, dims[0]/2,
            -dims[1]/2, dims[1]/2,
            -dims[2]/2, dims[2]/2
        )
        out_file = self.out_dir / 'relax_structure.dump'
        self._write_dump(centered, box, out_file)
        return out_file

    def _build_replica(self, reps):
        """
        Crea coordenadas únicas de la red a partir del cell base. 
        reps: tupla de enteros (nx,ny,nz).
        """
        if self.structure_type == 'fcc':
            base = np.array([
                [0,0,0],
                [0.5,0.5,0],
                [0.5,0,0.5],
                [0,0.5,0.5]
            ]) * self.lattice

        elif self.structure_type == 'bcc':
            base = np.array([
                [0,0,0],
                [0.5,0.5,0.5]
            ]) * self.lattice

        else:
            raise ValueError(f"Tipo no soportado: {self.structure_type}")

        reps = np.array(reps)
        dims = reps * self.lattice
        coords = []
        for i in range(reps[0]):
            for j in range(reps[1]):
                for k in range(reps[2]):
                    disp = np.array([i, j, k]) * self.lattice
                    for p in base:
                        coords.append(p + disp)
        coords = np.mod(np.array(coords), dims)
        unique = np.unique(np.round(coords, 6), axis=0)
        return unique, dims

    def _write_dump(self, coords, box, out_file: Path):
        """
        Escribe el archivo LAMMPS dump.
        """
        with out_file.open('w') as f:
            f.write("ITEM: TIMESTEP\n0\n")
            f.write(f"ITEM: NUMBER OF ATOMS\n{len(coords)}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"{box[0]} {box[1]}\n")
            f.write(f"{box[2]} {box[3]}\n")
            f.write(f"{box[4]} {box[5]}\n")
            f.write("ITEM: ATOMS id type x y z\n")
            for i, (x, y, z) in enumerate(coords, start=1):
                f.write(f"{i} 1 {x:.6f} {y:.6f} {z:.6f}\n")
