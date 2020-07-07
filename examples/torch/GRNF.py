from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from grnf.torch import GRNF as GRNF_layer

class GRNF(nn.Module):
    
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        # GRNF mapping
        self.grnf_layer = GRNF_layer(channels=int(config.num_random_features),
                                     in_node_channels=dim_features,
                                     in_edge_channels=0,
                                     hidden_features=int(config.num_hidden_features),
                                     order_2_prc=.67,
                                     center_embedding=False,
                                     normalize_basis=False,
                                     normalize=False)
        
        self.dense_layers = nn.ModuleList()
        in_feat = int(config.num_random_features)
        out_feat = int(config.num_hidden_neurons)
        for _ in range(config.num_dense_layers-1):
            self.dense_layers.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            in_feat = out_feat
        self.dense_layers.append(nn.Linear(in_features=in_feat, out_features=dim_target))

    def forward(self, data):
        psi = self.grnf_layer(data=data)
        x = psi
        for lin in self.dense_layers:
            x = lin(F.relu(x))
        
        return x
    
    def reset_parameters(self):
        self.grnf_layer.reset_parameters()
        self.out_layer.reset_parameters()
        for lin in self.dense_layers:
            lin.reset_parameters()
        return super().reset_parameters()
