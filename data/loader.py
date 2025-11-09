"""
Data loading functions for GlassVAE project.
Based on EGNN_15_OPTIMIZE.py
"""
import numpy as np


def load_pos(input_path):
    """
    Load atomic positions and types from LAMMPS dump file.
    Returns separate positions and atom types arrays.
    """
    with open(input_path, 'r') as f:
        content = f.read().split('ITEM: TIMESTEP\n')[1:]
    
    positions, atom_types = [], []
    for chunk in content:
        parts = chunk.split('\n')
        n_atoms = int(parts[2].strip())
        pos_start = 8
        pos_data = parts[pos_start:pos_start+n_atoms]
        positions.append(np.array([list(map(float, p.split()[2:5])) for p in pos_data]))
        atom_types.append(np.array([int(p.split()[1]) - 1 for p in pos_data]))
    
    return np.array(positions, dtype=np.float32), np.array(atom_types)


def open_energy(file_path):
    """
    Load energy values from file.
    """
    with open(file_path, 'r') as f:
        return np.array([float(line.split()[-1]) for line in f])

