

import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_geometric.nn import MetaLayer
from torch_geometric.nn import GATv2Conv, global_mean_pool, global_add_pool, GCN2Conv
from .layers import MLPBlock
from .model import Encoder 



# Simple Edge model to be used in MetaLayer
class EdgeModel(nn.Module):
    def __init__(self, in_channels):
        super(EdgeModel, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_channels*3, in_channels), nn.BatchNorm1d(in_channels), nn.SiLU())

    def forward(self, src, dest, edge_attr, u, batch):
        out = torch.cat([src, dest, edge_attr], dim=1)
        return self.mlp(out)

# Node model that performs an aggregating operation over nearby nodes 
class NodeModel(nn.Module):
    def __init__(self, in_channels):
        super(NodeModel, self).__init__()
        self.mlp1 = nn.Sequential(nn.Linear(in_channels*2, in_channels), nn.BatchNorm1d(in_channels), nn.SiLU())
        self.mlp2 = nn.Sequential(nn.Linear(in_channels*2, in_channels), nn.BatchNorm1d(in_channels), nn.SiLU())

    def forward(self, x, edge_index, edge_attr, u, batch):
        row, col = edge_index
        # Grab input node features
        out = torch.cat([x[row], edge_attr], dim=1)
        # Encode 
        out = self.mlp1(out)
        # Average to connected neighbors
        out = scatter_mean(out, col, dim=0, dim_size=x.size(0))
        # gather final representation
        out = torch.cat([x, out], dim=1)
        return self.mlp2(out)

# Residual GNN Layer
class ResLayer(nn.Module):
    def __init__(self, in_channels, dropout=0.10):
        super(ResLayer, self).__init__()
        self.projectDown_node = nn.Linear(in_channels, in_channels//4)
        self.projectDown_edge = nn.Linear(in_channels, in_channels//4)
        self.bn1_node = nn.BatchNorm1d(in_channels//4)
        self.bn1_edge = nn.BatchNorm1d(in_channels//4)
        
        #Experimenting with different layers
        # self.conv = MetaLayer(edge_model=EdgeModel(in_channels//4), node_model=NodeModel(in_channels//4), global_model=None)
        # self.conv = GATv2Conv(in_channels//4, in_channels//4, edge_dim=in_channels//4, heads=4, concat=False, add_self_loops=False)
        
        self.conv = GCN2Conv(channels=in_channels//4, alpha=0.3678)
                
        self.projectUp_node = nn.Linear(in_channels//4, in_channels)
        self.projectUp_edge = nn.Linear(in_channels//4, in_channels)
        self.dropout = nn.Dropout(p=dropout)
        self.bn2_node = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_node.weight)
        self.bn2_edge = nn.BatchNorm1d(in_channels)
        nn.init.zeros_(self.bn2_edge.weight)
    
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        #Encode features in down protection
        h_node = F.silu(self.bn1_node(self.projectDown_node(x)))
        h_edge = F.silu(self.bn1_edge(self.projectDown_edge(edge_attr)))
        # Convolve over node and edge features
        h_node = self.conv(h_node, h_node, edge_index)
        
        # Residual Connections and up projections
        h_node = self.dropout(self.bn2_node(self.projectUp_node(h_node)))
        data.x = F.silu(h_node + x)
        
        h_edge = self.dropout(self.bn2_edge(self.projectUp_edge(h_edge))) 
        data.edge_attr = F.silu(h_edge + edge_attr)
        
        return data

## Main ligand model
class LigandModel(nn.Module):
    
    def __init__(self, in_channels, edge_features=6, hidden_dim = 128, residual_layers = 4, mlp_layers = 3, key_dim = 1024, dropout_rate = 0.15):
        super(LigandModel, self).__init__()
        
        # Embedding layers
        self.node_embedding = nn.Linear(in_channels, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # Convolution layers followed by residual block
        self.conv1 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv2 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        self.conv3 = MetaLayer(edge_model=EdgeModel(hidden_dim), node_model=NodeModel(hidden_dim), global_model=None)
        reslayers = [ResLayer(hidden_dim, dropout_rate) for _ in range(residual_layers)]
        self.resnet = nn.Sequential(*reslayers)
        
        # Encoder and MLPs, this encoder is trained on its own, but still against the morgan fingerprints
        self.encoder = Encoder(key_dim, hidden_dim)
        self.energy_mlp = MLPBlock(hidden_dim*2, 4, mlp_layers, nn.SiLU(), dropout_rate)
        self.discriminiator = MLPBlock(hidden_dim*2, 4, mlp_layers*2, nn.SiLU(), dropout_rate)
    
        
    def forward(self, data, fp):
        # Embed features into latent space 
        data.x = self.node_embedding(data.x)
        data.edge_attr = self.edge_embedding(data.edge_attr)
        # Convolution layers
        data.x, data.edge_attr, _ = self.conv1(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data.x = F.dropout(data.x, p=0.15, training=self.training)
        data.x, data.edge_attr, _ = self.conv2(data.x, data.edge_index, data.edge_attr, None, data.batch)
        data.edge_attr = F.dropout(data.edge_attr, p=0.15, training=self.training)
        data.x, data.edge_attr, _ = self.conv3(data.x, data.edge_index, data.edge_attr, None, data.batch)
        # Resnet layers
        data = self.resnet(data)
        # Compress Graph into [B, N, Hidden_dim]
        x_pool = global_add_pool(data.x, data.batch)
        # Encode fingerprints into [B, N, Hidden_dim]
        fp_encode = F.silu(self.encoder(fp))

        # Concatenate representations and pass through MLPs
        comb_x = torch.cat([x_pool, fp_encode], dim=1)
        
        # Output energy and class predictions
        energy = F.relu(self.energy_mlp(comb_x))
        # Class predictions are not passed through activation since we use BCEWithLogitsLoss as the loss function
        cls_pred = self.discriminiator(comb_x)
        
        return energy, cls_pred        
        
        
