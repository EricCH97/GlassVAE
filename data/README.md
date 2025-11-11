# Data Format and Usage Guide

This folder contains data loading utilities and dataset classes for the GlassVAE project. This guide explains the expected data format and how to use the provided functions.

## Data Structure

### Directory Organization

The data should be organized in the following structure:

```
Data/
├── 700K/
│   ├── Positions_700K
│   └── EAM_Energies_7.6Ang
├── 720K/
│   ├── Positions_720K
│   └── EAM_Energies_7.6Ang
├── 840K/
│   ├── Positions_840K
│   └── EAM_Energies_7.6Ang
└── ...
```

Each temperature folder contains:
- **Positions file**: LAMMPS dump format with atomic positions and types
- **Energy file**: Text file with energy values (one per line)

### Position File Format (LAMMPS Dump)

The position files follow the LAMMPS dump format. Each timestep contains:

```
ITEM: TIMESTEP
0
ITEM: NUMBER OF ATOMS
108
ITEM: BOX BOUNDS pp pp pp
-2.0775111078035036e-01 1.2207751110770650e+01
-2.0775111078035036e-01 1.2207751110770650e+01
-2.0775111078035036e-01 1.2207751110770650e+01
ITEM: ATOMS id type x y z
1 1 4.84142 7.2738 0.854691
2 1 4.27418 1.14499 2.74375
3 1 10.7947 9.88495 3.37315
...
```

**Key Points:**
- Each timestep starts with `ITEM: TIMESTEP`
- Atom data format: `id type x y z`
- Atom types are 1-indexed in the file (will be converted to 0-indexed)
- Multiple timesteps can be in the same file

**What the loader extracts:**
- **Positions**: `[num_samples, num_atoms, 3]` - x, y, z coordinates
- **Atom Types**: `[num_samples, num_atoms]` - atom type indices (0-indexed)

### Energy File Format

The energy files are simple text files with one energy value per line:

```
   1 -523.5073954
   2 -522.3699188
   3 -524.6955381
   4 -522.1616669
   5 -522.0364618
```

**Format:**
- Each line contains: `index energy_value`
- The loader extracts the last column (energy value)
- Returns: `[num_samples]` array of energy values

# Data Used for this work

Related training data can be find through this link:
https://drive.google.com/drive/folders/12vbQczkV9EXxY1x9f0fhGWw9orYQITGi?usp=sharing


