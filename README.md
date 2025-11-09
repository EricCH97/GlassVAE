# GlassVAE

Variational Autoencoder (VAE) for Metallic Glass Structures using Graph Neural Networks.

## Project Structure

```
GlassVAE/
├── data/                    # Data loading and dataset modules
│   ├── __init__.py
│   ├── loader.py           # Functions for loading position and energy data
│   └── dataset.py          # SiO2Dataset class for PyTorch Geometric
├── models/                  # Model definitions
│   ├── __init__.py
│   ├── egnn.py             # E(n) Equivariant Graph Neural Network components
│   └── vae.py              # Variational Autoencoder model
├── utils/                   # Utility functions
│   ├── __init__.py
│   └── helpers.py          # Helper functions (segment operations, edge building)
├── losses.py               # Loss functions (VAE loss, Chamfer loss, etc.)
├── metrics.py              # Evaluation metrics (RMSE, R2, MAE)
├── train.py                # Training and evaluation functions
├── main.py                 # Main execution script
└── README.md               # This file
```

## Module Descriptions

### Data Module (`data/`)
- **loader.py**: Contains functions to load atomic positions and energy data from files
  - `load_pos()`: Loads atomic positions from LAMMPS dump format
  - `open_energy()`: Loads energy data from text files
- **dataset.py**: PyTorch Geometric dataset class for glass structures
  - `SiO2Dataset`: Custom dataset class for SiO2 glass structures

### Models Module (`models/`)
- **egnn.py**: E(n) Equivariant Graph Neural Network implementation
  - `E_GCL`: E(n) Equivariant Convolutional Layer
  - `EGNN`: Complete EGNN model
- **vae.py**: Variational Autoencoder for glass structures
  - `SiO2VAE`: VAE model with encoder, decoder, and prediction heads

### Utils Module (`utils/`)
- **helpers.py**: Utility functions
  - `unsorted_segment_sum()`: Segment sum operation
  - `unsorted_segment_mean()`: Segment mean operation
  - `build_radius_edges()`: Build edges based on distance cutoff

### Loss Functions (`losses.py`)
- `stoichiometry_loss()`: Enforces 1:1 stoichiometry constraint
- `chamfer_loss()`: Computes Chamfer distance between predicted and target positions
- `vae_loss()`: Combined VAE loss with reconstruction and regularization terms

### Metrics (`metrics.py`)
- `calculate_metrics()`: Computes RMSE, R2, and MAE metrics

### Training (`train.py`)
- `train_epoch()`: Training function for one epoch
- `evaluate()`: Evaluation function for validation set

### Main Script (`main.py`)
- Main execution script that orchestrates data loading, model training, and evaluation

## Usage

Run the main script from the GlassVAE directory:

```bash
cd GlassVAE
python main.py
```

## Dependencies

- PyTorch
- PyTorch Geometric
- NumPy
- scikit-learn

## Notes

- The project expects data files in `NewData/700K/` directory
- Model hyperparameters can be adjusted in `main.py`
- Anomaly detection is enabled for debugging (can be disabled in `main.py`)

