"""
Variational Autoencoder (VAE) model for glass structures.
Based on EGNN_15_OPTIMIZE.py - uses MessagePassing instead of custom EGNN.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import global_add_pool, MessagePassing

# Global constant
CUTOFF = 7.6


class EdgeConvBlock(nn.Module):
    """Simple MLP block with residual connection for processing edge features."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(out_dim, out_dim)
        )
        self.skip = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()

    def forward(self, x):
        return self.mlp(x) + self.skip(x)


class EdgeFeatureEncoder(nn.Module):
    """
    Processes raw edge attributes (dx, dy, dz, distance) and pools the features to a graph-level embedding.
    """
    def __init__(self, edge_dim=4, hidden_dim=64, num_layers=3):
        super().__init__()
        layers = []
        in_dim = edge_dim
        for _ in range(num_layers):
            layers.append(EdgeConvBlock(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.net = nn.Sequential(*layers)

    def forward(self, data):
        dist_only = data.edge_attr  # [num_edges, 4]
        edge_emb = self.net(dist_only)
        i = data.edge_index[0]
        edge_batch = data.batch[i]
        pooled = global_add_pool(edge_emb, edge_batch)
        return pooled


class GNNConvLayer(MessagePassing):
    """GNN layer using MessagePassing from PyTorch Geometric."""
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='mean')
        self.mlp = nn.Sequential(
            nn.Linear(in_channels*2 + 4, out_channels),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Linear(out_channels, out_channels)
        )

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index=edge_index, x=x, edge_attr=edge_attr)
        return out

    def message(self, x_i, x_j, edge_attr):
        msg_in = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(msg_in)


class NodeEncoder(nn.Module):
    """Encoder that maps nodes to latent space."""
    def __init__(self, node_dim, hidden_dim, latent_dim, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        in_dim = node_dim
        for _ in range(num_layers):
            self.layers.append(GNNConvLayer(in_dim, hidden_dim))
            in_dim = hidden_dim
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
        x_pooled = global_add_pool(x, batch)
        x_pooled = self.bn(x_pooled)
        mu = self.fc_mu(x_pooled)
        logvar = self.fc_logvar(x_pooled)
        return mu, logvar, x


class EdgeDecoder(nn.Module):
    """Decoder that predicts edge distances from latent representations."""
    def __init__(self, latent_dim, edge_hidden_dim):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(latent_dim * 2 + 4, edge_hidden_dim),
            nn.BatchNorm1d(edge_hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(edge_hidden_dim, 1)
        )

    def forward(self, z, edge_index, edge_attr):
        src, dst = edge_index
        batch_size = z.size(0)
        num_nodes = z.size(1) if len(z.shape) > 2 else 1
        
        # Reshape z if it's batched
        if len(z.shape) > 2:
            z = z.view(-1, z.size(-1))  # [batch_size * num_nodes, latent_dim]
        
        # Calculate batch indices for each edge
        edge_batch = src // num_nodes
        
        # Ensure batch indices are within bounds
        edge_batch = torch.clamp(edge_batch, 0, batch_size - 1)
        
        # Calculate offsets for each batch
        batch_offsets = torch.arange(0, batch_size, device=z.device) * num_nodes
        
        # Get the correct offset for each edge based on its batch
        edge_offsets = batch_offsets[edge_batch]
        
        # Calculate local indices within each batch
        src_local = src % num_nodes
        dst_local = dst % num_nodes
        
        # Add offsets to get global indices
        src_idx = src_local + edge_offsets
        dst_idx = dst_local + edge_offsets
        
        # Get source and destination embeddings
        z_src = z[src_idx]
        z_dst = z[dst_idx]
        
        # Concatenate and predict
        edge_input = torch.cat([z_src, z_dst, edge_attr], dim=1)
        edge_pred = self.edge_mlp(edge_input)
        return edge_pred


class Decoder(nn.Module):
    """Decoder that reconstructs nodes, edges, and positions from latent space."""
    def __init__(self, latent_dim, hidden_dim, num_types, num_nodes, edge_hidden_dim):
        super().__init__()
        self.num_types = num_types
        self.num_nodes = num_nodes
        self.max_dist = CUTOFF
        
        self.node_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_nodes * num_types)
        )
        
        self.edge_decoder = EdgeDecoder(latent_dim, edge_hidden_dim)
        
        self.pos_fc = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, num_nodes * 3)
        )

    def forward(self, z, edge_index, edge_attr):
        node_logits = self.node_fc(z).view(z.size(0), self.num_nodes, self.num_types)
        edge_pred = self.edge_decoder(z, edge_index, edge_attr)
        pos_pred = self.pos_fc(z).view(z.size(0), self.num_nodes, 3)
        return node_logits, edge_pred, pos_pred


class CombinedVAE(nn.Module):
    """Combined VAE model with node encoder, edge encoder, and decoder."""
    def __init__(self, node_dim, hidden_dim, latent_dim, num_nodes, edge_dim=4, edge_hidden=64):
        super().__init__()
        self.encoder = NodeEncoder(node_dim, hidden_dim, latent_dim, num_layers=2)
        self.decoder = Decoder(latent_dim, hidden_dim, node_dim, num_nodes, edge_hidden)
        self.edge_encoder = EdgeFeatureEncoder(edge_dim=edge_dim, hidden_dim=edge_hidden, num_layers=3)
        combined_dim = latent_dim + edge_hidden
        self.energy_head = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, 1)
        )
        for layer in self.energy_head:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, nonlinearity='leaky_relu')
                nn.init.constant_(layer.bias, 0.0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, data):
        mu, logvar, _ = self.encoder(data)
        z = self.reparameterize(mu, logvar)
        node_pred, edge_pred, pos_pred = self.decoder(z, data.edge_index, data.edge_attr)
        edge_emb = self.edge_encoder(data)
        combined = torch.cat([mu, edge_emb], dim=-1)
        energy = self.energy_head(combined)
        return node_pred, edge_pred, energy, mu, logvar, pos_pred

