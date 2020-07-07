from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from grnf import GRNF2_layer

class ModuleWithPrecomputedRepresentations(nn.Module):
    can_precompute_representations = True
    _has_precomputed_representations = False

    def precompute_representations(self, dataset, batch_size=64, verbose=True, **kwargs):
        """ Exposed method """
        from torch_geometric.data import Batch

        tbar = tqdm(range(len(dataset.data)), desc="precompute representations", disable=not verbose)
        for b in range(len(dataset.data)//batch_size+1):
            ga, gb = b*batch_size, min([(b+1)*batch_size, len(dataset.data)])
            batch = Batch.from_data_list(dataset.data[ga: gb])

            prec_repr = self._precompute_representations(data=batch, **kwargs)
            assert prec_repr.shape[0] == gb-ga

            for i in range(gb-ga):
                tbar.update()
                dataset.data[ga + i].prec_repr = prec_repr[i].unsqueeze(0)
                dataset.data[ga + i].keys.append("prec_repr")

        tbar.close()

        self._has_precomputed_representations = True
        return dataset

    def _precompute_representations(self, data, **kwargs):
        """ To be implemented by the subclass """
        raise NotImplementedError

    def reset_parameters(self):
        self._has_precomputed_representations = False
        return super().reset_parameters()


class GRNF(ModuleWithPrecomputedRepresentations):
    
    def __init__(self, dim_features, dim_target, config):
        super().__init__()

        self.grnf_layer = GRNF2_layer(in_node_features=dim_features,
                                      in_edge_features=0,
                                      out_features=config.num_random_features,
                                      hidden_features=config.num_hidden_features)
          
        self.dense_layers = nn.ModuleList()
        in_feat = config.num_random_features
        out_feat = config.num_hidden_neurons
        for _ in range(config.num_dense_layers-1):
            self.dense_layers.append(nn.Linear(in_features=in_feat, out_features=out_feat))
            in_feat = out_feat
        self.dense_layers.append(nn.Linear(in_features=in_feat, out_features=dim_target))

    def forward(self, data):
       
        if self._has_precomputed_representations:
            psi = data.prec_repr
        else:
            psi = self.grnf_layer(data=data)

        x = psi
        for lin in self.dense_layers:
            x = lin(F.relu(x))
        
        return x
    
    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        self.grnf_layer.to(*args, **kwargs)
        for lin in self.dense_layers:
            lin.to(*args, **kwargs)
        # self.out_layer.to(*args, **kwargs)
        return self    

    def _precompute_representations(self, data):
        return self.grnf_layer(data=data)

    def reset_parameters(self):
        self.grnf_layer.reset_parameters()
        self.out_layer.reset_parameters()
        for lin in self.dense_layers:
            lin.reset_parameters()
        return super().reset_parameters()
