"""
Variational Autoencoder (VAE) model for glass structures.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_add_pool
from .egnn import EGNN


class SiO2VAE(nn.Module):
    """VAE model for SiO2 glass structures."""
    
    def __init__(self, node_dim=1, hidden_dim=128, latent_dim=64, 
                 n_layers=4, cutoff=3.0):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Encoder
        self.encoder = EGNN(node_dim, hidden_dim, hidden_dim, 
                           n_layers=n_layers, cutoff=cutoff, normalize=True)  # Enabled normalization
        
        # Latent space
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder = EGNN(latent_dim, hidden_dim, hidden_dim, 
                           n_layers=n_layers, cutoff=cutoff, normalize=True)  # Enabled normalization
        
        self.coord_init = nn.Parameter(torch.randn(1, 3))  # Initialize once for the entire batch
        
        # For per-atom classification (e.g., Si vs O)
        self.atom_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Energy prediction
        self.energy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.SiLU(),
            nn.Linear(hidden_dim//2, 1)
        )

    def encode(self, data):
        h, _ = self.encoder(data.x, data.pos, data.batch)
        # Using global_add_pool as per original script
        h_pool = global_add_pool(h, data.batch)
        return self.fc_mu(h_pool), self.fc_logvar(h_pool)

    def decode(self, z, data):
        z_expanded = z[data.batch]  # [N, latent_dim]
        
        # Initialize coordinates with smaller noise for stability
        coords_init = self.coord_init.repeat(z_expanded.size(0), 1)
        noise = torch.randn_like(coords_init) * 0.01  # Reduced noise from 0.1 to 0.01
        coords_init = coords_init + noise

        # Decode to get updated node features and coordinates
        h_out, coords_out = self.decoder(z_expanded, coords_init, data.batch)
        
        # 1) Atom type prediction
        atom_pred = torch.sigmoid(self.atom_head(h_out))  # => [N,1]
        
        # 2) Energy prediction
        pooled = global_mean_pool(h_out, data.batch)      # => [B, hidden_dim]
        energy_pred = self.energy_head(pooled)            # => [B,1]
        
        return atom_pred, coords_out, energy_pred

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, logvar = self.encode(data)
        z = self.reparameterize(mu, logvar)
        atom_pred, coords_pred, energy_pred = self.decode(z, data)
        return atom_pred, coords_pred, energy_pred, mu, logvar

