"""
Graph construction functions for GlassVAE project.
Based on EGNN_15_OPTIMIZE.py
"""
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import Data

# Global constant
CUTOFF = 7.6


def build_graph(types: Tensor, positions: Tensor, box: Tensor) -> Data:
    """
    Construct a PyG Data object with periodic boundary conditions:
      - node features: one-hot encoded atom types
      - edges: computed from periodic displacements (dx, dy, dz, distance)
    """
    n_nodes = positions.size(0)
    disp = positions.unsqueeze(1) - positions.unsqueeze(0)
    
    # Apply periodic boundary conditions
    box_tensor = box.unsqueeze(0).unsqueeze(0)
    disp = torch.remainder(disp + box_tensor/2, box_tensor) - box_tensor/2
    
    # Compute distances and ignore self-edges
    distances = torch.norm(disp, dim=-1)
    distances.fill_diagonal_(float('inf'))
    
    edge_mask = distances < CUTOFF
    senders, receivers = torch.nonzero(edge_mask, as_tuple=True)
    
    num_types = int(types.max()) + 1
    node_features = F.one_hot(types.long(), num_classes=num_types).float()
    
    # Concatenate displacement and distance information for edge attributes
    edge_attr = torch.cat([
        disp[senders, receivers],
        distances[senders, receivers].unsqueeze(-1)
    ], dim=-1)
    
    data = Data(
        x=node_features,
        edge_index=torch.stack([senders, receivers]),
        edge_attr=edge_attr
    )
    return data


def create_graph_data(pos, energy, types, box):
    """
    Build a PyG Data object for one sample.
      - y: energy (scaled)
      - node_target: true atom type (as integer labels)
      - edge_target: ground-truth edge attributes (including distance)
      - pos_true: original 3D positions for RDF loss
    """
    graph = build_graph(types, pos, box)
    graph.y = torch.tensor([energy], dtype=torch.float32)
    graph.node_target = torch.argmax(graph.x, dim=1)
    graph.edge_target = graph.edge_attr.clone()
    graph.pos_true = pos  # store true positions for later RDF loss computation
    return graph

