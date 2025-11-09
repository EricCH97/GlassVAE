"""
E(n) Equivariant Graph Neural Network (EGNN) components.
"""
import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.helpers import unsorted_segment_sum, unsorted_segment_mean, build_radius_edges


class E_GCL(nn.Module):
    """E(n) Equivariant Convolutional Layer"""
    
    def __init__(self, input_nf, output_nf, hidden_nf, edges_in_d=0, 
                 act_fn=nn.SiLU(), residual=True, attention=False, 
                 normalize=False, coords_agg='mean', tanh=False):
        super().__init__()
        input_edge = input_nf * 2
        self.residual = residual
        self.attention = attention
        self.normalize = normalize
        self.coords_agg = coords_agg
        self.tanh = tanh
        self.epsilon = 1e-8

        # Edge MLP
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + 1 + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn)

        # Node MLP
        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf))

        # Coordinate MLP
        coord_mlp = [
            nn.Linear(hidden_nf, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, 1, bias=False)
        ]
        if tanh:
            coord_mlp.append(nn.Tanh())
        self.coord_mlp = nn.Sequential(*coord_mlp)

        # Attention
        if attention:
            self.att_mlp = nn.Sequential(nn.Linear(hidden_nf, 1), nn.Sigmoid())

    def edge_model(self, source, target, radial, edge_attr):
        # Concatenate node features with geometric information
        if edge_attr is None:  # No additional edge attributes
            out = torch.cat([source, target, radial], dim=1)
        else:  # Include edge attributes if available
            out = torch.cat([source, target, radial, edge_attr], dim=1)
            
        # Process through edge MLP
        out = self.edge_mlp(out)
        
        # Apply attention mechanism if enabled
        if self.attention:
            att = self.att_mlp(out)  # Compute attention weights [E, 1]
            out = out * att  # Element-wise multiplication
            
        return out

    def node_model(self, x, edge_index, edge_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        out = torch.cat([x, agg], dim=1)
        out = self.node_mlp(out)
        return x + out if self.residual else out

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        return coord + agg

    def forward(self, h, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, dim=1, keepdim=True)
        
        if self.normalize:
            norm = torch.sqrt(radial).detach() + self.epsilon
            coord_diff = coord_diff / norm

        edge_feat = self.edge_model(h[row], h[col], radial, None)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h = self.node_model(h, edge_index, edge_feat)
        return h, coord


class EGNN(nn.Module):
    """E(n) Equivariant Graph Neural Network"""
    
    def __init__(self, in_node_nf, hidden_nf, out_node_nf, n_layers=4, 
                 device='cuda', cutoff=3.0, **kwargs):
        super().__init__()
        self.build_radius_edges = build_radius_edges
        
        self.embedding = nn.Linear(in_node_nf, hidden_nf)
        self.layers = nn.ModuleList([
            E_GCL(hidden_nf, hidden_nf, hidden_nf, **kwargs) for _ in range(n_layers)
        ])
        self.out_layer = nn.Linear(hidden_nf, out_node_nf)
        self.cutoff = cutoff
        self.device = device

    def forward(self, h, x, batch):
        # Construct edges based on radius
        edge_index = self.build_radius_edges(x, batch, self.cutoff)
        
        h = self.embedding(h)
        for layer in self.layers:
            h, x = layer(h, edge_index, x)
        return self.out_layer(h), x

