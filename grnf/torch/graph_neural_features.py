from grnf import utils

import torch
from torch.nn import Parameter
from torch.distributions import Binomial
from torch_scatter import scatter_add

def get_activation(activation_object):
    r""" Parse the activation """
    if activation_object is None:
        return lambda x: x
    elif callable(activation_object):
        return activation_object
    else:
        return torch.nn.functional.__dict__.get(activation_object)
    
def get_initializer(initializer_object):
    r""" Parse the initializer """
    if callable(initializer_object):
        return initializer_object
    else:
        return torch.nn.init.__dict__.get(initializer_object)

def get_normalization_factors(normalize, num_nodes, batch, device):
    r""" Replicates the normalization factor to (batch_size, 1) (num_tot_nodes|batch, 1) """
    fact_b = torch.ones(1, device=device)
    fact_n = torch.ones(1, device=device)
    if normalize:
        fact_b = fact_b / num_nodes
        fact_n = fact_b[batch].unsqueeze(1)
        fact_b = fact_b.unsqueeze(1)
    return fact_n, fact_b

def count_parameters(model, verbose=False):
    r""" Counts the number of trainable and not trainable parameters """
    al = 0
    tr = 0
    for p in list(model.parameters()):
        nn = 1
        for s in list(p.size()):
            nn = nn * s
        al += nn
        if p.requires_grad:
            tr += nn
    if verbose:
        print("--------------------------------------------")
        print("Num. parameters                     {}".format(al))
        print("Num. trainable parameters           {}".format(tr))
        print("Num. non-trainable parameters       {}".format(al-tr))
        print("--------------------------------------------")
    return tr, al - tr

class _NeuralFeatures(torch.nn.Module):
    r"""
    Base class for graph neural features of a predefined hidden tensor order
    ([Zambon et al. 2020]([https://arxiv.org/abs/1909.03790)).
    """
    def __init__(self,
                 channels,  # out_feature
                 in_node_channels,
                 in_edge_channels,
                 hidden_features=None,
                 activation=None,
                 hidden_activation="relu",
                 hidden_tensor_order=1,
                 use_bias=True,
                 kernel_initializer=torch.nn.init.normal_,
                 bias_initializer=torch.nn.init.normal_,
                 normalize=True,
                 center_embedding=True,
                 **kwargs):

        super().__init__(**kwargs)
        
        # --- Generic param ---------
        self.activation = get_activation(activation)
        self.hidden_activation = get_activation(hidden_activation)
        self.use_bias = use_bias
        self.kernel_initializer = get_initializer(kernel_initializer)
        self.bias_initializer = get_initializer(bias_initializer)
        self.center_embedding = center_embedding

        # --- GRNF param ---------
        self.num_hidden_features = hidden_features
        self.num_grnf = channels
        self._normalize = normalize
        self.hidden_tensor_order = hidden_tensor_order
        
        # build
        self.num_node_features = in_node_channels
        self.num_edge_features = in_edge_channels

        if self.num_hidden_features is None:
            self.num_hidden_features = max([1, 2 * (self.num_node_features + self.num_edge_features)])

        # set parameters
        eq_k_sh, eq_b_sh = utils.param_shapes(k_in=2,
                                              k_out=self.hidden_tensor_order,
                                              feat_in=self.num_in_features,
                                              feat_out=self.num_hidden_features,
                                              feat_rand=self.num_grnf)

        in_k_sh, in_b_sh = utils.param_shapes(k_in=self.hidden_tensor_order,
                                              k_out=0,
                                              feat_in=self.num_hidden_features,
                                              feat_out=1,
                                              feat_rand=self.num_grnf)

        self.kernel_equiv = Parameter(torch.Tensor(*eq_k_sh))
        self.kernel_inv = Parameter(torch.Tensor(*in_k_sh))
        if self.use_bias:
            self.bias_equiv = Parameter(torch.Tensor(*eq_b_sh))
            self.bias_inv = Parameter(torch.Tensor(*in_b_sh))
        else:
            self.register_parameter('bias_equiv', None)
            self.register_parameter('bias_inv', None)

        if self.center_embedding:
            self.zerograph = Parameter(torch.Tensor((1, self.num_grnf)), requires_grad=False)
        else:
            self.register_parameter('zerograph', None)
            
        self.reset_parameters()

    @property
    def _device(self):
        return self.kernel_equiv.device
    
    @property
    def num_in_features(self):
        """ When there are no edge features, then the adjacency matrix is considered. """
        return self.num_node_features + max([1, self.num_edge_features])

    def reset_parameters(self):
        self.kernel_initializer(self.kernel_equiv)
        self.kernel_initializer(self.kernel_inv)
        if self.use_bias:
            self.bias_initializer(self.bias_equiv)
            self.bias_initializer(self.bias_inv)
        if self.center_embedding:
            # precompute centering
            zerograph = self.activation(self.compute_zerograph())
            self.zerograph = Parameter(zerograph, requires_grad=False)

    def forward(self, data):
        # first convenient input representation, a list with shapes
        #     - (num_tot_nodes|batch, num_in_feat) 
        #     - (batch_size, num_in_feat).
        repr_compact, num_nodes, fact_norm = self.parse_input(data=data)
        
        # compute neural features psi
        psi = self.compute_neural_features(repr_compact=repr_compact, data=data,
                                           num_nodes=num_nodes, fact_norm=fact_norm)
        psi = self.activation(psi)

        if self.center_embedding:
            psi.sub_(self.zerograph)
            
        return psi

    def compute_zerograph(self):
        r"""
        Computes the GRNF associated with a ``null'' graph `0_g` with
        a single node, no self-loops, zero edge and node attributes.
        
        The resulting representation serves as center to have a kernel induced
        by the distance; see Section 5 in [Zambon et al. 2020](https://arxiv.org/abs/1909.03790)
        for further details.
        """
        if self.use_bias:
            repr_eq = self.hidden_activation(self.bias_equiv[..., 0, :])
            inv_basis = repr_eq
            repr_inv = torch.einsum("bmh, mdh -> bmd", inv_basis, self.kernel_inv[:, 0, ...])
            repr_inv.add_(self.bias_inv)
        else:
            repr_inv = torch.zeros((1, self.num_grnf, 1), device=self._device)
        return  repr_inv[..., 0]
    
    def parse_input(self, data):
        r"""
        Create a first compact and convenient representation of the graphs; see also Appendix A
        in [Maron et al. 2019](https://arxiv.org/abs/1812.09902).

        The output representation is a list containing
        ```
            repr_a = [
                diag_part:     (num_tot_nodes|batch, num_in_feat)
                sum_diag_part:          (batch_size, num_in_feat)
                sum_of_rows:   (num_tot_nodes|batch, num_in_feat)
                sum_of_cols:   (num_tot_nodes|batch, num_in_feat)
                sum_all:                (batch_size, num_in_feat)
            ]
        ```
        """

        # Auxiliary vars
        diag_bidx = data.edge_index[0] == data.edge_index[1]  # nodes with self loops (diagonal)
        num_tot_nodes = data.batch.shape[0]                   # total number of nodes
        if self.num_node_features > 0:
            _x = data.x.type(torch.float)
        else:
            _x = torch.ones((num_tot_nodes, 0), device=self._device)
        if self.num_edge_features > 0:
            _edge_attr = data.edge_attr.type(torch.float)
        else:
            _edge_attr = torch.ones((data.edge_index.shape[1], 1), device=self._device)
        
        # normalization
        num_nodes = scatter_add(src=torch.ones(data.batch.shape, device=self._device, dtype=torch.long), index=data.batch, dim=0)
        fact_n, fact_b = get_normalization_factors(self._normalize, num_nodes, data.batch, device=self._device)
        batch_size = len(num_nodes)

        # Compact representation
        #diag_part (num_tot_nodes|batch, num_in_feat)
        diag_part = torch.zeros((num_tot_nodes, self.num_in_features), device=self._device)
        diag_part[:, :self.num_node_features] = _x
        diag_part[data.edge_index[0, diag_bidx], self.num_node_features:] = _edge_attr[diag_bidx]
        #sum_diag_part (batch_size, num_in_feat)
        sum_diag_part = scatter_add(src=diag_part, index=data.batch, dim=0, dim_size=batch_size).mul_(fact_b)
        #sum_of_rows (num_tot_nodes|batch, num_in_feat)
        sum_of_rows = torch.cat([_x, scatter_add(src=_edge_attr, index=data.edge_index[0], dim=0, dim_size=num_tot_nodes)],
                                dim=1).mul_(fact_n)
        #sum_of_cols (num_tot_nodes|batch, num_in_feat)
        sum_of_cols = torch.cat([_x, scatter_add(src=_edge_attr, index=data.edge_index[1], dim=0, dim_size=num_tot_nodes)],
                                dim=1).mul_(fact_n)

        #sum_all = (batch_size, num_in_feat)
        sum_all = scatter_add(src=sum_of_cols, index=data.batch, dim=0, dim_size=batch_size).mul_(fact_b).mul_(fact_b)

        repr_compact = {
            "diag_part": diag_part,
            "sum_diag_part": sum_diag_part,
            "sum_of_rows": sum_of_rows,
            "sum_of_cols": sum_of_cols,
            "sum_all": sum_all,
        }
        
        return repr_compact, num_nodes, fact_b

    def get_grnf_weights(self):
        w = {}
        w["kernel_equiv"] = self.kernel_equiv.numpy()
        w["kernel_inv"] = self.kernel_inv.numpy()
        if self.use_bias:
            w["bias_equiv"] = self.bias_equiv.numpy()
            w["bias_inv"] = self.bias_inv.numpy()
        return w
    
    def set_grnf_weights(self, w):
        assert self.kernel_equiv.shape[1:] == w["kernel_equiv"].shape[1:]
        assert self.kernel_inv.shape[1:] == w["kernel_inv"].shape[1:]
        self.kernel_equiv = Parameter(torch.tensor(w["kernel_equiv"]))
        self.kernel_inv = Parameter(torch.tensor(w["kernel_inv"]))
        self.num_grnf = w["kernel_inv"].shape[0]
        if self.use_bias:
            assert self.bias_equiv.shape[2:] == w["bias_equiv"].shape[2:]
            assert self.bias_inv.shape[2:] == w["bias_inv"].shape[2:]
            self.bias_equiv = Parameter(torch.tensor(w["bias_equiv"]))
            self.bias_inv = Parameter(torch.tensor(w["bias_inv"]))

        return self


class NeuralFeatures1(_NeuralFeatures):
    r"""
    Extends `grnf.torch._NeuralFeatures` and implements GRNF all with hidden
    tensor order equal to 1. 
    """
    
    def __init__(self, channels, **kwargs):
        self.name = "NeuFeat1"
        assert kwargs.pop("hidden_tensor_order", 1) == 1
        super().__init__(channels=channels, hidden_tensor_order=1, **kwargs)
        assert self.hidden_tensor_order == 1
    
    def compute_neural_features(self, repr_compact, data, num_nodes, fact_norm):
        r"""
        The input data comes in the `repr_compact`, that is
        ```
            repr_compact = [diag_part, sum_diag_part, sum_of_rows, sum_of_cols, sum_all],
        ```
        which have shapes `(num_tot_nodes|batch, num_in_feat)` and `(batch_size, num_in_feat)`.

        The equivariant representation is constructed from compact one, as a linear 
        combination of Bell(k_hid+2) components, resulting in `repr_eq` with shape
        `(num_tot_nodes|batch, num_neural_feat, num_hidden_feat)`.
        The computation employs and auxiliary tensor `repr_eq_sum` with shape 
        `(batch_size, num_neural_feat, num_hidden_feat)`

        The output is an invariant representation `repr_inv` with shape
        `(batch_size, num_neural_feat)`. Notice that it should have been
        `(batch_size, num_neural_feat, bell(k_out), num_out_feat)`, however,
        `k_out=0`, `bell(0)=1` and `num_out_feat is set to 1.

        **Notation:**

        - `n`, `l`: num_nodes
        - `f`: node_in_feat + edge_feat
        - `m`: num neural features
        - `h`: hidden feature dimension
        - `m`: num neural features
        - `d`: output feature dimension (=1)
        """

        # --- Equivariant part  -----------------------------

        #linear
        # op1 - (123) - extract diag
        repr_eq = torch.einsum("nf, mhf -> nmh", repr_compact["diag_part"], self.kernel_equiv[:, 0, ...])
        # op2 - (123) + (12)(3) - tile sum of diag part
        repr_eq_sum = torch.einsum("bf, mhf -> bmh", repr_compact["sum_diag_part"], self.kernel_equiv[:, 1, ...])
        # op3 - (123) + (13)(2) - place sum of row i in element i
        repr_eq.add_(torch.einsum("nf, mhf -> nmh", repr_compact["sum_of_rows"], self.kernel_equiv[:, 2, ...]))
        # op4 - (123) + (23)(1) - place sum of col i in element i
        repr_eq.add_(torch.einsum("nf, mhf -> nmh", repr_compact["sum_of_cols"], self.kernel_equiv[:, 3, ...]))
        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        repr_eq_sum.add_(torch.einsum("bf, mhf -> bmh", repr_compact["sum_all"], self.kernel_equiv[:, 4, ...]))
        
        repr_eq.add_(repr_eq_sum[data.batch, ...])

        #bias
        if self.use_bias:
            repr_eq.add_(self.bias_equiv[..., 0, :])

        #activation
        repr_eq = self.hidden_activation(repr_eq)
        
        # --- Invariant part ----------------------------------

        #linear (with normalization)
        inv_basis = scatter_add(src=repr_eq, index=data.batch, dim=0, dim_size=len(num_nodes)).mul_(fact_norm.unsqueeze(1))
        repr_inv = torch.einsum("bmh, midh -> bmd", inv_basis, self.kernel_inv)

        #bias
        if self.use_bias:
            repr_inv.add_(self.bias_inv)

        return repr_inv[..., 0]


class NeuralFeatures2(_NeuralFeatures):
    r"""
    Extends `grnf.torch._NeuralFeatures` and implements GRNF all with hidden
    tensor order equal to 2. 
    """

    def __init__(self, channels, **kwargs):
        self.name = "NeuFeat2"
        assert kwargs.pop("hidden_tensor_order", 2) == 2
        super().__init__(channels=channels, hidden_tensor_order=2, **kwargs)
        assert self.hidden_tensor_order == 2

    def compute_neural_features(self, repr_compact, data, num_nodes, fact_norm):
        r"""
        The input data comes in the `repr_compact`, that is
        ```
            repr_compact = [diag_part, sum_diag_part, sum_of_rows, sum_of_cols, sum_all],
        ```
        which have shapes `(num_tot_nodes|batch, num_in_feat)` and `(batch_size, num_in_feat)`.

        The equivariant representation is constructed from compact one, as a linear
        combination of Bell(k_hid+2) components, resulting in `repr_eq` with shape
        `(num_tot_nodes|batch, num_tot_nodes|batch, num_neural_feat, num_hidden_feat)`.
        The computation employs a collection of auxiliary tensors to be set or repeated
        along the diagonal, the rows or the columns:
        ```
            repr_eq_set_diag: with shape (num_neural_feat, num_tot_nodes|batch, num_hidden_feat)
                which results from:
                    (diag(rep), f) x (m, e=1, d, f) -> (m, n, n, d)
                    expanded: (n, n, f) x (m, e=1, d, f) -> (m, n, n, d)
                    compact:   (n|b, f) x (m, e=1, d, f) -> (m, n|b, d)
            repr_eq_rep_diag: with shape (num_neural_feat, batch_size, num_hidden_feat)
                which results from:
                    (rep*eye, f) x (m, e=1, d, f) -> (m, n, n, d)
                    expanded: (n, n, f) x (m, e=1, d, f) -> (m, n, n, d)
                    compact:     (b, f) x (m, e=1, d, f) -> (m, b, d)
            repr_eq_rep_row: with shape (num_neural_feat, num_tot_nodes|batch, num_hidden_feat)
                which results from:
                    (rep(n, 1).dot(ones(1, n)), f) x (m, e=1, d, f) -> (m, n, n, d)
                    expanded: (n, n, f) x (m, e=1, d, f) -> (m, n, n, d)
                    compact:   (n|b, f) x (m, e=1, d, f) -> (m, n|b, d)
            repr_eq_rep_col: with shape (num_neural_feat, num_tot_nodes|batch, num_hidden_feat)
                which results from:
                    (ones(n, 1).dot(rep(1, n)), f) x (m, e=1, d, f) -> (m, n, n, d)
                    expanded: (n, n, f) x (m, e=1, d, f) -> (m, n, n, d)
                    compact:   (n|b, f) x (m, e=1, d, f) -> (m, n|b, d)
            repr_eq_x: with shape (num_neural_feat, num_tot_nodes|batch, num_hidden_feat)
            repr_eq_E: with shape (num_neural_feat, num_tot_edges|batch, num_hidden_feat)
        ```
        The output is an invariant representation `repr_inv` with shape
        `(batch_size, num_neural_feat)`. Notice that it should have been
        `(batch_size, num_neural_feat, bell(k_out), num_out_feat)`, however,
        `k_out=0`, `bell(0)=1` and `num_out_feat is set to 1.

        **Notation**
        
        - `n`, `l`: num_nodes
        - `f`: node_in_feat + edge_feat
        - `m`: num neural features
        - `h`: hidden feature dimension
        - `m`: num neural features
        - `d`: output feature dimension (=1)
        """

        # --- Equivariant part  -----------------------------

        if self.num_node_features > 0:
            _x = data.x.type(torch.float)
        else:
            _x = torch.ones((num_nodes.sum(), 0), device=self._device)

        if self.num_edge_features > 0:
            _edge_attr = data.edge_attr.type(torch.float)
        else:
            _edge_attr = torch.ones((data.edge_index.shape[1], 1), device=self._device)

        #linear
        # op1 - (1234) - extract diag
        repr_eq_set_diag = torch.einsum("nf, mdf -> nmd", repr_compact["diag_part"], self.kernel_equiv[:, 0, ...])
        # op2 - (1234) + (12)(34) - place sum of diag on diag
        repr_eq_rep_diag = torch.einsum("bf, mdf -> bmd", repr_compact["sum_diag_part"], self.kernel_equiv[:, 1, ...])
        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        repr_eq_set_diag.add_(torch.einsum("nf, mdf -> nmd", repr_compact["sum_of_rows"], self.kernel_equiv[:, 2, ...]))
        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        repr_eq_set_diag.add_(torch.einsum("nf, mdf -> nmd", repr_compact["sum_of_cols"], self.kernel_equiv[:, 3, ...]))
        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        repr_eq_rep_diag.add_(torch.einsum("bf, mdf -> bmd", repr_compact["sum_all"], self.kernel_equiv[:, 4, ...]))
        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        repr_eq_rep_row = torch.einsum("nf, mdf -> nmd", repr_compact["sum_of_cols"], self.kernel_equiv[:, 5, ...])
        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        repr_eq_rep_row.add_(torch.einsum("nf, mdf -> nmd", repr_compact["sum_of_rows"], self.kernel_equiv[:, 6, ...]))
        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        repr_eq_rep_col = torch.einsum("nf, mdf -> nmd", repr_compact["sum_of_cols"], self.kernel_equiv[:, 7, ...])
        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        repr_eq_rep_col.add_(torch.einsum("nf, mdf -> nmd", repr_compact["sum_of_rows"], self.kernel_equiv[:, 8, ...]))
        # op10 - (1234) + (14)(23) - identity
        if self.num_node_features > 0:
            repr_eq_set_diag.add_(torch.einsum("nf, mdf -> nmd", _x, self.kernel_equiv[:, 9, :, :self.num_node_features]))
        repr_eq_E = torch.einsum("af, mdf -> amd", _edge_attr, self.kernel_equiv[:, 9, :, self.num_node_features:])
        # op11 - (1234) + (13)(24) - transpose
        if self.num_node_features > 0:
            repr_eq_set_diag.add_(torch.einsum("nf, mdf -> nmd", _x, self.kernel_equiv[:, 10, :, :self.num_node_features]))
        repr_eq_Et = torch.einsum("af, mdf -> amd", _edge_attr, self.kernel_equiv[:, 10, :, self.num_node_features:])
        # op12 - (1234) + (234)(1) - place ii element in row i
        repr_eq_rep_row.add_(torch.einsum("nf, mdf -> nmd", repr_compact["diag_part"], self.kernel_equiv[:, 11, ...]))
        # op13 - (1234) + (134)(2) - place ii element in col i
        repr_eq_rep_col.add_(torch.einsum("nf, mdf -> nmd", repr_compact["diag_part"], self.kernel_equiv[:, 12, ...]))
        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        repr_eq_rep_all = torch.einsum("bf, mdf -> bmd", repr_compact["sum_diag_part"], self.kernel_equiv[:, 13, ...])
        # op15 - sum of all ops - place sum of all entries in all entries
        repr_eq_rep_all.add_(torch.einsum("bf, mdf -> bmd", repr_compact["sum_all"], self.kernel_equiv[:, 14, ...]))

        repr_eq_set_diag.add_(repr_eq_rep_diag[data.batch, ...])
        repr_eq_rep_col.add_(repr_eq_rep_all[data.batch, ...])

        if self.use_bias:
            # bias can be directly addded here
            repr_eq_rep_col.add_(self.bias_equiv[..., 1, :])
            repr_eq_set_diag.add_(self.bias_equiv[..., 0, :]).sub_(self.bias_equiv[..., 1, :])

        # auxiliary variables
        n_, m_, b_, d_ = data.batch.shape[0], self.num_grnf, num_nodes.shape[0], self.num_hidden_features
        arange_ = torch.arange(n_, dtype=torch.long, device=self._device)  # .unsqueeze_(0)
        batch_edges = torch.nonzero(data.batch.unsqueeze(0) == data.batch.unsqueeze(1), as_tuple=False).T 
        repr_eq_rep_full = repr_eq_rep_col[batch_edges[0]] + repr_eq_rep_row[batch_edges[1]]
        
        repr_eq_indices = torch.cat([
            arange_.repeat(2, 1),                                #diagonal
            batch_edges,                                         #sum of rows and cols
            data.edge_index,                                     #edge_attr
            data.edge_index[torch.LongTensor([1, 0])],           #edge_attr transpose
        ], dim=1)

        repr_eq_values = torch.cat([
            repr_eq_set_diag,
            repr_eq_rep_full,
            repr_eq_E,
            repr_eq_Et
        ], dim=0)

        repr_eq = torch.sparse.FloatTensor(
            repr_eq_indices,
            repr_eq_values,
            torch.Size([n_, n_, m_, d_])).coalesce()
       
        #activation
        # ... moved in invariant part

        # --- Invariant part ----------------------------------

        #linear (with normalization)
        diag_idx = repr_eq.indices()[0] == repr_eq.indices()[1]
        offd_idx = torch.logical_not(diag_idx)
        batch_size = len(num_nodes)
        inv_basis_diag = scatter_add(
            src=self.hidden_activation(repr_eq.values()[diag_idx]),
            index=data.batch, dim=0, dim_size=batch_size).mul_(fact_norm.unsqueeze(1))
        inv_basis_offd = scatter_add(
            src=self.hidden_activation(repr_eq.values()[offd_idx]),
            index=data.batch[repr_eq.indices()[0][offd_idx]],
            dim=0, dim_size=batch_size).mul_(fact_norm.mul(fact_norm).unsqueeze(1))
        
        # inv_basis = torch.cat([inv_basis_diag, inv_basis_offd], dim=3)
        inv_basis = torch.stack([inv_basis_diag, inv_basis_offd], dim=3)
        repr_inv = torch.einsum("bmhi, midh -> bmd", inv_basis, self.kernel_inv)

        #bias
        if self.use_bias:
            repr_inv += self.bias_inv

        return repr_inv[..., 0]
    
class GraphRandomNeuralFeatures(torch.nn.Module):
    r"""
    Graph Neural Random Features (GRNF)
    [Zambon et al. 2020](https://arxiv.org/abs/1909.03790).
    GRNF maps a graph $g$ with $n$ nodes to an $m$-dimensional vector $\psi$.
    
    Graph $g$ is firstly represented an order-2 tensor
    $$
        A_g=[\diag(X)|E] \in \R^{n^2 \times f+s},
    $$
    where $f, s$ are the dimension of the node and edge features, respectively.
    Subsequently $A_g$ is mapped to $m$ (hidden) tensors
    $$
        [T_i \in \R^{n^{k_i}} \times h} for i in range(m)]
    $$
    by means of node-permutation equivariant maps and, finally, to a vector
    $$
        \psi = \psi(g) \in \R^m,
    $$
    via a invariant maps.
    
    **Input**

    - Binary adjacency matrix of shape `(n, n)`;
    - Node features of shape `(n, f)` (optional);
    - Edge features of shape `(n, s)` (optional);
    
    **Output**

    - GRNF: Graph-level representation of the input graph of shape `(m,)`.

    **Arguments**

    - `channels`: integer, number of output channels, i.e., number `m` of neural
        graph random features;
    - `in_node_channels`: integer, dimension `f` of the node features;
    - `in_edge_channels`: integer, dimension `s` of the edge features;
    - `hidden_features`: integer, dimension `h` of the features in the hidden tensor
        representation;
    - `hidden_activation`: activation function to use between the equivariant and
        invariant layers;
    - `activation`: activation function to use on top of the GRNF embedding (but
        before `1/sqrt(m)` normalization);
    - `order_2_prc`: ratio of features with hidden tensor order 2 out of the total;
    - `use_bias`: bool, add a bias vector to the output;
    - `kernel_initializer`: initializer for the weights;
    - `bias_initializer`: initializer for the bias vector;
    - `normalize`: boolean, if `True` the output is multiplied bu `1/sqrt(m)`;
    - `normalize_basis`: boolean, if `True` the linear invariant and equivariant
        maps are normalized;
    - `center_embedding`: boolean, if `True` the embedding of the graph with a
        single node, and 0 features,
        is subtracted from the `GRNF(g)`;
    - `trainable`: boolean, if `True`, then beta is a trainable parameter.
    
    
    Specifically, input data come in PyG [COO format](https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#data-handling-of-graphs),
    so `data` has the following attributes:
    
    - `data.x`: Node feature matrix with shape `(num_tot_nodes|batch, f)`
    - `data.edge_index`: Edge list with shape `(2, num_tot_edges)`
    - `data.edge_attr`: Edge feature matrix with shape `(num_tot_edges, s)`
    - `data.batch`: Column vector (num_tot_nodes) mapping each node to its
        respective graph in the batch

    The notation `num_tot_nodes|batch` stresses that they keep the batch
    subdivision of `data.x`. If no edge feature is present, then the adjacency
    matrix is used as edge features and `num_in_feat = max(1, f+s)`.
    The current version supports orders $k \in \{1, 2\}.

    """
    
    def __init__(self,
                 channels,  # out_feature
                 in_node_channels,
                 in_edge_channels,
                 hidden_features=None,
                 activation=None,
                 hidden_activation="relu",
                 order_2_prc=.7,
                 use_bias=True,
                 kernel_initializer=torch.nn.init.normal_,
                 bias_initializer=torch.nn.init.normal_,
                 normalize=True,
                 normalize_basis=True,
                 trainable=False,
                 center_embedding=True,
                 **kwargs):

        self.name = "GRNF"
        super().__init__(**kwargs)

        self.num_grnf = channels
        self.order_2_prc = order_2_prc
        
        self.trainable = trainable
        
        self._pars = {}  # parameters to pass to NeuralFeatures classes
        self._pars["activation"] = activation
        self._pars["hidden_activation"] = hidden_activation
        self._pars["use_bias"] = use_bias
        self._pars["kernel_initializer"] = kernel_initializer
        self._pars["bias_initializer"] = bias_initializer
        
        self._pars["hidden_features"] = hidden_features
        self._normalize = normalize
        self._pars["normalize"] = normalize_basis
        self._pars["center_embedding"] = center_embedding
    
        # Build
        self.num_node_features = in_node_channels
        self._pars["in_node_channels"] = self.num_node_features
        self.num_edge_features = in_edge_channels
        self._pars["in_edge_channels"] = self.num_edge_features

        if self.order_2_prc == 0:
            num_order1, num_order2 = self.num_grnf, 0
        elif self.order_2_prc == 1:
            num_order1, num_order2 = 0, self.num_grnf
        else:
            num_order2 = int(Binomial(total_count=self.num_grnf, probs=self.order_2_prc).sample())
            num_order2 = max([min([self.num_grnf - 1, num_order2]), 1])
            num_order1 = self.num_grnf - num_order2
        assert (num_order1 + num_order2) == self.num_grnf

        self.psi_order_1 = NeuralFeatures1(num_order1, **self._pars) if num_order1 > 0 else None
        self.psi_order_2 = NeuralFeatures2(num_order2, **self._pars) if num_order2 > 0 else None

        self.set_trainable()

    def set_trainable(self):
        if self.psi_order_1 is not None:
            for p in self.psi_order_1.parameters():
                p.requires_grad = self.trainable
        if self.psi_order_2 is not None:
            for p in self.psi_order_2.parameters():
                p.requires_grad = self.trainable

    def forward(self, data):
        if self.psi_order_1 is None:
            return self.psi_order_2(data)
        if self.psi_order_2 is None:
            return self.psi_order_1(data)
        psi1 = self.psi_order_1(data)
        psi2 = self.psi_order_2(data)
        psi = torch.cat((psi1, psi2), dim=1)
       
        # reweight to approximate distance and kernel
        if self._normalize:
            psi /= torch.sqrt(torch.tensor(self.num_grnf, dtype=torch.float32))

        return psi

    def get_grnf_weights(self):
        w = {}
        if self.psi_order_1 is not None:
            w["k1"] = self.psi_order_1.get_grnf_weights()
        if self.psi_order_2 is not None:
            w["k2"] = self.psi_order_2.get_grnf_weights()
        return w
    
    def set_grnf_weights(self, w):
        if self.psi_order_1 is not None:
            self.psi_order_1.set_grnf_weights(w["k1"])
        else:
            assert not hasattr(w, "k1")
        if self.psi_order_2 is not None:
            self.psi_order_2.set_grnf_weights(w["k2"])
        else:
            assert not hasattr(w, "k2")
            
        self.set_trainable()
        return self


class ChunkGraphRandomNeuralFeatures(torch.nn.Module):
    r"""
    Wrapper to GraphRandomNeuralFeatures.
    
    When the number of features is very large, it is convenient to
    divide the computation into bunches of features to save memory.
    
    The size of the chunk is passed with `chunk_size` attribute.
    """
    
    def __init__(self,
                 channels,  # out_feature
                 chunk_size=1000,
                 normalize=True,
                 **kwargs):
    
        self._grnf_kwargs = {}
        for k in [
            "in_node_channels",
            "in_edge_channels",
            "hidden_features",
            "activation",
            "hidden_activation",
            "order_2_prc",
            "use_bias",
            "kernel_initializer",
            "bias_initializer",
            "normalize",
            "normalize_basis",
            "trainable",
            "center_embedding"]:
            try:
                self._grnf_kwargs[k] = kwargs.pop(k)
            except:
                pass

        self.name = "rGRNF"
        super().__init__(**kwargs)
    
        self.num_grnf = channels
        self.chunk_size = chunk_size
        self.num_rags = self.num_grnf//self.rag_size
        self._normalize = normalize,
        self.grnf_rags = torch.nn.ModuleList(
            [GraphRandomNeuralFeatures(channels=self.chunk_size, normalize=False, **self._grnf_kwargs)
             for _ in range(self.num_rags)])
        reminder = channels % self.chunk_size
        if reminder > 0:
            self.grnf_chunks.append(GraphRandomNeuralFeatures(channels=reminder, normalize=False, **self._grnf_kwargs))
            self.num_rags += 1
            
    def forward(self, data, verbose=False):
        
        from tqdm import tqdm
        psi_list = []
        for cgrnf in tqdm(self.grnf_chunks,
                          disable=not verbose,
                          desc="cGRNF.forward"):
            psi_list.append(cgrnf(data))
        psi = torch.cat(psi_list, dim=1)

        # reweight to approximate distance and kernel
        if self._normalize:
            psi /= torch.sqrt(torch.tensor(self.num_grnf, dtype=torch.float32))
        
        return psi

    def get_grnf_weights(self):
        w = []
        for grnf in self.grnf_rags:
            w.append(grnf.get_grnf_weights())
        return w

    def set_grnf_weights(self, w):
        for wi, i in enumerate(w):
            self.grnf_rags[i].set_grnf_weights(wi)
    
        self.set_trainable()
        return self
