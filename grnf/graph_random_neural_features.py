import torch
from torch.nn import Linear
from torch.nn.functional import relu, log_softmax
from torch.distributions import Binomial
from torch_scatter import scatter_add

from tqdm import tqdm

# List of Bell's number
# _bell = [None, 1, 2, 5, 15, 52, 203, 877, 4140, 21147, 115975, 678570, 4213597, 27644437, 190899322, 1382958545, 10480142147, 82864869804, 682076806159, 5832742205057, 51724158235372, 474869816156751, 4506715738447323, 44152005855084346, 445958869294805289, 4638590332229999353, 49631246523618756274]
_bell = [None, 1, 2, 5, 15]
def bell(i):
    return _bell[i]

LINEAR, BIAS = 0, 1
def _init_random_param(k_in, k_out, feat_in, feat_out, feat_rand, sigma, mu):
    """ Generate the weights """
    Gamma_lin = bell(k_in + k_out)
    Gamma_bias = bell(k_out)
    lin = torch.randn(feat_rand, Gamma_lin,   feat_out, feat_in)
    if Gamma_bias is not None:
        bias = torch.randn(feat_rand, Gamma_bias, feat_out)
    else:
        bias = torch.randn(feat_rand,             feat_out)
    ret = [0, 0]
    ret[LINEAR] =  lin * sigma + mu
    ret[BIAS]   = bias * sigma + mu
    return ret

def _eff_impl_basis(data, graph_idx, num_node_features, num_edge_features, device):
    """ See Maron, Haggai, et al. Invariant and equivariant graph networks. ICLR 2019 """
    # node slice
    batch_idx = (data.batch == graph_idx).nonzero()
    ns = [int(batch_idx[0]), int(batch_idx[-1]) + 1]
    num_nodes = ns[1] - ns[0]
    # edges slice
    es = data.__slices__["edge_index"][graph_idx: graph_idx + 2]
    num_edges = es[1] - es[0]
        
    # gloc stands for the "local" representation of the graph, i.e., with indexing startingr from 0.
    gloc = {}
    # node attributes
    if num_node_features > 0:
        gloc['x'] = data.x[ns[0]: ns[1], :].to(device=device)
    gloc['edge_index'] = data.edge_index[:, es[0]: es[1]].to(device=device) - ns[0]
    # edges attributes
    if num_edge_features > 0:
        gloc['edge_attr'] = data.edge_attr[es[0]: es[1], :].to(device=device)
    else:
        num_edge_features = 1
        gloc['edge_attr'] = torch.ones((num_edges, 1)).to(device=device)
    
    diag_lidx = gloc['edge_index'][0] == gloc['edge_index'][1]
    rows_lidx = gloc['edge_index'][0].unique()
    cols_lidx = gloc['edge_index'][1].unique()
        
    # diag_part = tf.matrix_diag_part(inputs)  # N x D x m
    diag_part = torch.zeros((num_nodes, num_node_features + num_edge_features), device=device)  # (_, N**k_hid, Feat_node + Feat_edge)
    if num_node_features > 0:
        diag_part[:, :num_node_features] = gloc['x']
    diag_part[gloc['edge_index'][0, diag_lidx], num_node_features:] = gloc['edge_attr'][diag_lidx]
    # sum_diag_part = tf.reduce_sum(diag_part, axis=2, keepdims=True)  # N x D x 1
    sum_diag_part = diag_part.sum(axis=0)  # (_, _, Feat_node + Feat_edge)
    # sum_of_rows = tf.reduce_sum(inputs, axis=3)  # N x D x m
    sum_of_rows = torch.zeros((num_nodes, num_node_features + num_edge_features), device=device)  # (_, N**k_hid, Feat_node + Feat_edge)
    if num_node_features > 0:
        sum_of_rows[:, :num_node_features] = gloc['x']
    # sum_of_rows[:, feat_node:] = scatter_add(data.edge_attr[data.adj[:, 1]],
    #                                                           data.adj[:, 1])  # N x D x m
    sum_of_rows[:rows_lidx[-1] + 1, num_node_features:] = scatter_add(src=gloc['edge_attr'],
                                                                      index=gloc['edge_index'][0], dim=0)

    # sum_of_cols = tf.reduce_sum(inputs, axis=2)  # N x D x m
    sum_of_cols = torch.zeros((num_nodes, num_node_features + num_edge_features), device=device)  # (_, N**k_hid, Feat_node + Feat_edge)
    if num_node_features > 0:
        sum_of_cols[:, :num_node_features] = gloc['x']
    sum_of_cols[:cols_lidx[-1] + 1, num_node_features:] = scatter_add(src=gloc['edge_attr'],
                                                                      index=gloc['edge_index'][1], dim=0)
    
    # sum_all = tf.reduce_sum(inputs, axis=(2, 3))  # N x D
    sum_all = sum_of_cols.sum(axis=0)  # (_, _, Feat_node + Feat_edge)
    
    return num_nodes, gloc, diag_part, sum_diag_part, sum_of_rows, sum_of_cols, sum_all

def _eff_impl_k1(eff_impl_bas_to_unpack, num_node_features, num_edge_features, normalize, device):
    """ See Maron, Haggai, et al. Invariant and equivariant graph networks. ICLR 2019 """
    k_in, k_out = 2, 1
    Gamma = bell(k_in + k_out)
    
    num_nodes, g, diag_part, sum_diag_part, sum_of_rows, sum_of_cols, sum_all = eff_impl_bas_to_unpack
    
    if num_edge_features == 0: num_edge_features = 1
    
    repr = torch.zeros((Gamma, num_nodes, num_node_features + num_edge_features), device=device)  # (Gamma_eq, N**k_hid, Feat_node + Feat_edge)
    # op1 - (123) - extract diag
    # op1 = diag_part  # N x D x m
    repr[0] = diag_part
    # op2 - (123) + (12)(3) - tile sum of diag part
    # op2 = tf.tile(sum_diag_part, [1, 1, dim])  # N x D x m
    # op2 = sum_diag_part.repeat(1, 1, num_nodes)
    repr[1] = sum_diag_part.repeat(num_nodes, 1)
    # op3 - (123) + (13)(2) - place sum of row i in element i
    # op3 = sum_of_rows  # N x D x m
    repr[2] = sum_of_rows
    # op4 - (123) + (23)(1) - place sum of col i in element i
    # op4 = sum_of_cols  # N x D x m
    repr[3] = sum_of_cols
    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    # op5 = tf.tile(tf.expand_dims(sum_all, axis=2), [1, 1, dim])  # N x D x m
    # op5 = sum_all.repeat(1, sum_of_cols.shape[0], num_nodes)
    repr[4] = sum_all.repeat(num_nodes, 1)
    
    # if normalization is not None:
    if normalize is not None or normalize == False:
        repr /= 1. * num_nodes
        repr[-1] /= 1. * num_nodes
    
    return repr

def _eff_impl_k2(eff_impl_bas_to_unpack, num_node_features, num_edge_features, normalize, device):
    """ See Maron, Haggai, et al. Invariant and equivariant graph networks. ICLR 2019 """
    k_in, k_out = 2, 2
    Gamma = bell(k_in + k_out)
    
    num_nodes, g, diag_part, sum_diag_part, sum_of_rows, sum_of_cols, sum_all = eff_impl_bas_to_unpack
    
    if num_edge_features == 0: num_edge_features = 1
    
    repr = torch.zeros((Gamma, num_nodes, num_nodes, num_node_features + num_edge_features), device=device)  # (Gamma_eq, N, N, Feat_node + Feat_edge)
    
    aran = torch.arange(num_nodes, device=device)
    # diag_part = tf.matrix_diag_part(inputs)   # N x D x m
    # sum_diag_part = tf.reduce_sum(diag_part, axis=2, keepdims=True)  # N x D x 1
    # sum_of_rows = tf.reduce_sum(inputs, axis=3)  # N x D x m
    # sum_of_cols = tf.reduce_sum(inputs, axis=2)  # N x D x m
    # sum_all = tf.reduce_sum(sum_of_rows, axis=2)  # N x D
    
    # op1 - (1234) - extract diag
    # op1 = tf.matrix_diag(diag_part)  # N x D x m x m
    repr[0, aran, aran] = diag_part
    
    # op2 - (1234) + (12)(34) - place sum of diag on diag
    # op2 = tf.matrix_diag(tf.tile(sum_diag_part, [1, 1, dim]))  # N x D x m x m
    repr[1, aran, aran] = sum_diag_part.repeat(num_nodes, 1)
    
    # op3 - (1234) + (123)(4) - place sum of row i on diag ii
    # op3 = tf.matrix_diag(sum_of_rows)  # N x D x m x m
    repr[2, aran, aran] = sum_of_rows
    
    # op4 - (1234) + (124)(3) - place sum of col i on diag ii
    # op4 = tf.matrix_diag(sum_of_cols)  # N x D x m x m
    repr[3, aran, aran] = sum_of_cols
    
    # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
    # op5 = tf.matrix_diag(tf.tile(tf.expand_dims(sum_all, axis=2), [1, 1, dim]))  # N x D x m x m
    repr[4, aran, aran] = sum_all.repeat(num_nodes, 1)
    
    # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
    # op6 = tf.tile(tf.expand_dims(sum_of_cols, axis=3), [1, 1, 1, dim])  # N x D x m x m
    repr[5] = sum_of_cols.unsqueeze(1).repeat(1, num_nodes, 1)
    
    # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
    # op7 = tf.tile(tf.expand_dims(sum_of_rows, axis=3), [1, 1, 1, dim])  # N x D x m x m
    repr[6] = sum_of_rows.unsqueeze(1).repeat(1, num_nodes, 1)
    
    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    # op8 = tf.tile(tf.expand_dims(sum_of_cols, axis=2), [1, 1, dim, 1])  # N x D x m x m
    repr[7] = sum_of_cols.unsqueeze(0).repeat(num_nodes, 1, 1)
    
    # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
    # op9 = tf.tile(tf.expand_dims(sum_of_rows, axis=2), [1, 1, dim, 1])  # N x D x m x m
    repr[8] = sum_of_rows.unsqueeze(0).repeat(num_nodes, 1, 1)
    
    # op10 - (1234) + (14)(23) - identity
    # op10 = inputs  # N x D x m x m
    # S = torch.sparse_coo_tensor(indices=g['edge_index'], values=g['edge_attr'],
    #                             size=[num_nodes, num_nodes, num_edge_features])
    # repr[9] = torch.zeros((num_nodes, num_nodes, num_node_features + num_edge_features))
    if hasattr(g, "edge_attr"):
        repr[9, g['edge_index'][0], g['edge_index'][1], num_node_features:] = g["edge_attr"]
    if hasattr(g, "x"):
        repr[9, aran, aran, :num_node_features] = g['x']
    
    # op11 - (1234) + (13)(24) - transpose
    # op11 = tf.transpose(inputs, [0, 1, 3, 2])  # N x D x m x m
    repr[10] = repr[9].transpose(0, 1)
    
    # op12 - (1234) + (234)(1) - place ii element in row i
    # op12 = tf.tile(tf.expand_dims(diag_part, axis=3), [1, 1, 1, dim])  # N x D x m x m
    repr[11] = diag_part.unsqueeze(1).repeat(1, num_nodes, 1)
    
    # op13 - (1234) + (134)(2) - place ii element in col i
    # op13 = tf.tile(tf.expand_dims(diag_part, axis=2), [1, 1, dim, 1])  # N x D x m x m
    repr[12] = diag_part.unsqueeze(0).repeat(num_nodes, 1, 1)
    
    # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
    # op14 = tf.tile(tf.expand_dims(sum_diag_part, axis=3), [1, 1, dim, dim])   # N x D x m x m
    repr[13] = sum_diag_part.repeat(num_nodes, num_nodes, 1)
    
    # op15 - sum of all ops - place sum of all entries in all entries
    # op15 = tf.tile(tf.expand_dims(tf.expand_dims(sum_all, axis=2), axis=3), [1, 1, dim, dim])  # N x D x m x m
    repr[14] = sum_all.repeat(num_nodes, num_nodes, 1)
    
    # if normalization is not None:
    if normalize is not None or normalize == False:
        # if normalization is 'inf':
        # op2 = tf.divide(op2, float_dim)
        # op3 = tf.divide(op3, float_dim)
        # op4 = tf.divide(op4, float_dim)
        # op5 = tf.divide(op5, float_dim**2)
        # op6 = tf.divide(op6, float_dim)
        # op7 = tf.divide(op7, float_dim)
        # op8 = tf.divide(op8, float_dim)
        # op9 = tf.divide(op9, float_dim)
        # op14 = tf.divide(op14, float_dim)
        # op15 = tf.divide(op15, float_dim**2)
        repr /= 1. * num_nodes
        repr[4] /= 1. * num_nodes
        repr[-1] /= 1. * num_nodes
    
    return repr


def _all_the_reminder(k, repr, param, act_hid, normalize, device):
    # Equivariant affine map
    # linear part
    # e: equiv. embedding dimension
    # n, l: num_nodes
    # f: node_feat + edge_feat
    # m: num random features
    # d: output (hidden) feature dimension
    num_nodes = repr.shape[1]
    
    if k == 1:
        eq_repr = torch.einsum("enf, medf -> mnd", repr,
                               param["equiv"][k][LINEAR])  # (Rand_feat, N, ..., N, Feat_hidden)
        eq_repr += param["equiv"][k][BIAS][:, 0, None, :]
    else:
        diag_el = torch.arange(num_nodes, device=device)
        eq_repr = torch.einsum("enlf, medf -> mnld", repr,
                               param["equiv"][k][LINEAR])  # (Rand_feat, N, ..., N, Feat_hidden)
        eq_repr[:, diag_el, diag_el, :] += param["equiv"][k][BIAS][:, 1, None, :]
        # (Num_rand_feat, N**k_hid, Feat_hidden)
    
    # Equivariant activation
    eq_repr = act_hid(eq_repr)  # (Rand_feat, N**k_hid, Feat_hidden)
    
    # --- invariant --------------------------------------------------------------------- #
    
    # Invariant repr
    if k == 1:
        inv_repr = torch.sum(eq_repr, axis=1).unsqueeze(1)  # (Num_rand_feat, Gamma_i=1, Feat_hid)
    else:
        inv_repr = torch.zeros(eq_repr.shape[0], bell(2), eq_repr.shape[-1]).to(device=device)
        inv_repr[:, 0, :] = eq_repr[:, diag_el, diag_el, :].sum(axis=1)
        inv_repr[:, 1, :] = eq_repr.sum(axis=[1, 2]) - inv_repr[:, 0, :]
    
    if normalize is not None or normalize == False:
        inv_repr /= 1. * num_nodes
        if k == 2:
            inv_repr[:, 1, :] /= 1. * num_nodes
    
    # Invariant affine map
    # linear part
    # m: num random features
    # i: inv. embedding dimension
    # f: hidden feature dimension
    # n: num_nodes^k
    # d: output feature dimension = 1
    psi = torch.einsum("mif, midf -> md", inv_repr, param["inv"][k][LINEAR])  # (Num_rand_feat, Feat_inv=1)
    psi += param["inv"][k][BIAS]
    
    return psi[:, 0]

def grnf_k12(data, graph_idx, num_node_features, num_edge_features,
             param, act_hid, normalize, device, zerograph=False):
    """
    Compute all the graph neural features with hidden tensor of order k_hid for the single graph in data.

    :param data:
        - edge_index (2, num_edges)
        - edge_attr (num_edges, feat_edge)
        - x (num_nodes, feat_node)
    """
    
    # --- equivariant --------------------------------------------------------------------- #
    # Given:
    #   gamma \in {1, ..., Bell(k_in + k_hid)}
    #   A.shape = (N, N, Feat_node | Feat_edge)
    #   E_gamma = (N**k_hid, N, N)
    #   alpha_gamma = (Feat_out, Feat_node | Feat_edge)
    #   Bias_eq = (N**k, Feat_out)
    # A single affine equivariant map is
    #   T = Sum_gamma { alpha_gamma * E_gamma . A }  +  Bias_eq
    #     = alpha . Sum_gamma { E_gamma . A } +  Bias_eq
    #     = alpha . R_eq  +  Bias_eq
    
    if zerograph:
        tot_feat = num_node_features + max([num_edge_features, 1])
        repr_k1 = torch.zeros((bell(2 + 1), 1, tot_feat), device=device)  # (Gamma_eq, N**k_hid, Feat_node + Feat_edge)
        repr_k2 = torch.zeros((bell(2 + 2), 1, 1, tot_feat), device=device)  # (Gamma_eq, N, N, Feat_node + Feat_edge)

    else:
        eff_impl_basis_to_unpack = _eff_impl_basis(data=data, graph_idx=graph_idx,
                                                   num_node_features=num_node_features, num_edge_features=num_edge_features,
                                                   device=device)
        
        repr_k1 = _eff_impl_k1(eff_impl_bas_to_unpack=eff_impl_basis_to_unpack,
                               num_node_features=num_node_features, num_edge_features=num_edge_features,
                               normalize=normalize, device=device)
        
        repr_k2 = _eff_impl_k2(eff_impl_bas_to_unpack=eff_impl_basis_to_unpack,
                               num_node_features=num_node_features, num_edge_features=num_edge_features,
                               normalize=normalize, device=device)

    psi_1 = _all_the_reminder(k=1, repr=repr_k1, param=param, act_hid=act_hid, normalize=normalize, device=device)
    psi_2 = _all_the_reminder(k=2, repr=repr_k2, param=param, act_hid=act_hid, normalize=normalize, device=device)

    return torch.cat([psi_2, psi_1], axis=0)

class GRNF2_layer(torch.nn.Module):
    _normalize = True
    _mu = 0.
    _sigma = 1.

    def __init__(self, in_node_features, in_edge_features, out_features,
                 hidden_features=None, activation=relu, **kwargs):
        super().__init__()
        # Each input graph would be a tensor
        #   A = (N, N, num_node_feat + num_edge_feat)
        # then mapped to hidden tensors
        #   T = (num_grnf, N, ..., N, num_hid_feat)
        # an activation is applied and, finally, to a vector
        #   (num_grnf,)
        self.num_node_features = in_node_features
        self.num_edge_features = in_edge_features
        self.num_hidden_features = hidden_features
        self.num_grnf = out_features
        self.activation = activation
        if hasattr(kwargs, "normalize"):
            self._normalize = kwargs["normalize"]
        # untrained parameters
        self.param = None
        
        # hyper-parameters
        self.hidden_tensor_orders = [1, 2]
        self.ratio_grnf_per_order = kwargs.get("ratio_grnf_per_order", {1: .67, 2: .33})
        if self.num_hidden_features is None:
            self.num_hidden_features = max([1, 2 * (self.num_node_features + self.num_edge_features)])
        
        self.reset_parameters()
        
        # check compliance
        assert self.num_grnf >= 2
        assert self.num_in_features >= 1
        assert sum(self.ratio_grnf_per_order.values()) == 1.
    
    @property
    def num_in_features(self):
        return self.num_node_features + max([1, self.num_edge_features])
        
    def reset_parameters(self):
        self.param = {}
        self.param["equiv"] = {}
        self.param["inv"] = {}
        
        # define number of random feature with each tensor order parameters
        n = int(Binomial(total_count=self.num_grnf, probs=self.ratio_grnf_per_order[1]).sample())
        n = max([min([self.num_grnf - 1, n]), 1])
        self._num_grnf_per_order = {1: n, 2: self.num_grnf - n}
        assert sum(self._num_grnf_per_order.values()) == self.num_grnf

        # generate random parameters
        pars = {"mu": self._mu, "sigma": self._sigma}
        # equivariant part
        pars["feat_in"] = self.num_in_features
        pars["feat_out"] = self.num_hidden_features
        self.param["equiv"][1] = _init_random_param(k_in=2, k_out=1, feat_rand=self._num_grnf_per_order[1], **pars)
        self.param["equiv"][2] = _init_random_param(k_in=2, k_out=2, feat_rand=self._num_grnf_per_order[2], **pars)
        # invariant part
        pars["feat_in"] = self.num_hidden_features
        pars["feat_out"] = 1
        self.param["inv"][1] = _init_random_param(k_in=1, k_out=0, feat_rand=self._num_grnf_per_order[1], **pars)
        self.param["inv"][2] = _init_random_param(k_in=2, k_out=0, feat_rand=self._num_grnf_per_order[2], **pars)

        return self
    
    # def graph_neural_features_layer(self, data, n_jobs=None, verbose=False):
    def forward(self, data, verbose=False, zerograph=False):
        """ Receives the batch and compute for every graph and every hidden tensor order the GRNF. """
        
        if zerograph:
            num_graphs = 1
        else:
            # get indices from data COO
            # node_range, edge_range = get_graph_ranges_from_coo(data)
            num_graphs = data.y.shape[0]
            data.edge_index.to(device=self._device)
            try:
                data.x.to(device=self._device)
            except:
                pass
            try:
                data.edge_attr.to(device=self._device)
            except:
                pass
        
        # output tensor
        psi = torch.zeros((num_graphs, self.num_grnf), device=self._device)  # , device=self.grnf_device)
        # single-graph processing
        for g in tqdm(range(num_graphs), desc="creating GRNF representations.", disable=not verbose):
            psi[g] = grnf_k12(data=data, graph_idx=g,
                              num_node_features=self.num_node_features, num_edge_features=self.num_edge_features,
                              param=self.param, act_hid=self.activation, normalize=self._normalize,
                              device=self._device, zerograph=zerograph)
        return psi

    def to(self, *args, **kwargs):
        self = super().to(*args, **kwargs)
        for k in [1, 2]:
            for typ in ["equiv", "inv"]:
                for par in [LINEAR, BIAS]:
                    self.param[typ][k][par] = self.param[typ][k][par].to(*args, **kwargs)
        self._device = kwargs.get("device", args[0])
        return self

    def get_zerograph_representation(self, *args, **kwargs):
        return self.forward(data=None, zerograph=True)