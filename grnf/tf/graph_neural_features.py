from .graph_neural_features_batch import NeuralFeatures1Batch, NeuralFeatures2Batch
from .graph_neural_features_disjoint import NeuralFeatures1Disjoint, NeuralFeatures2Disjoint

import tensorflow as tf
from tensorflow.keras.layers import Layer, Concatenate
from tensorflow.keras.backend import random_binomial
from spektral.layers import ops as spektral_ops

SPEKTRAL_DISJOINT = -1000

def get_spektral_mode(input_shape):
    r""" Identify the data [mode](https://graphneural.network/data/) """
    l = len(input_shape)
    if l == 4:
        assert len(input_shape[0]) == 2
        assert len(input_shape[1]) == 2
        assert len(input_shape[2]) == 2
        assert len(input_shape[3]) == 1
        return SPEKTRAL_DISJOINT, "DISJOINT", "XAEI"  # X (N|B, F),  A (2, E),  E (E, S),  I (N|B)
    elif l == 3:
        ndim0 = len(input_shape[0])  # X  |  X  |  A
        ndim1 = len(input_shape[1])  # A  |  A  |  E
        ndim2 = len(input_shape[2])  # E  |  I  |  I
        
        if ndim0 == 3 and ndim1 == 3 and ndim2 == 4:  # X (B, N, F),  A (B, N, N),  E (B, N, N, S)
            return spektral_ops.modes.BATCH, "BATCH", "XAE"
        elif ndim0 == 2 and ndim1 == 2 and ndim2 == 1:  # X (N|B, F),  A (2, E),  I (N|B)
            if input_shape[0][0] == input_shape[0][1]:
                pattern = "AEI"
            elif input_shape[1][0] == input_shape[1][1]:
                pattern = "XAI"
            else:
                raise NotImplementedError
            return SPEKTRAL_DISJOINT, "DISJOINT", pattern
        else:
            raise NotImplementedError
    elif l == 2:
        ndim0 = len(input_shape[0])  # X | A | A
        ndim1 = len(input_shape[1])  # A | E | I
        if ndim1 == 1:
            assert input_shape[0][0] == input_shape[0][1]
            return SPEKTRAL_DISJOINT, "DISJOINT", "AI"
        elif ndim1 == 3:
            assert input_shape[1][1] == input_shape[1][2]
            return spektral_ops.modes.BATCH, "BATCH", "XA"
        elif ndim1 == 4:
            assert input_shape[0][1] == input_shape[0][2]
            assert input_shape[1][1] == input_shape[1][2]
            return spektral_ops.modes.BATCH, "BATCH", "AE"
        else:
            raise NotImplementedError
    elif l == 1:
        assert input_shape[0][1] == input_shape[0][2]
        return spektral_ops.modes.BATCH, "BATCH", "A"  # A (B, N, N)
    else:
        raise NotImplementedError

class GraphRandomNeuralFeatures(Layer):
    r"""
    Graph Random Neural Features (GRNF)
    [Zambon et al. 2020](https://arxiv.org/abs/1909.03790).
    GRNF maps a graph $g$ with $n$ nodes to an $m$-dimensional vector $\psi$.
    
    **Mode**: disjoint, batch.
    
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
    
    If no edge feature is present, then the adjacency
    matrix is used as edge features and `num_in_feat = max(1, f+s)`.
    The current version supports orders $k \in \{1, 2\}.

    """
    
    def __init__(self,
                 channels,  # out_feature
                 hidden_features=None,
                 activation=None,
                 hidden_activation="relu",
                 order_2_prc=.7,
                 use_bias=True,
                 kernel_initializer=tf.random_normal_initializer(mean=0., stddev=1.),
                 bias_initializer=tf.random_normal_initializer(mean=0., stddev=1.),
                 normalize=True,
                 normalize_basis=True,
                 trainable=False,
                 center_embedding=True,
                 **kwargs):

        if "name" not in kwargs:
            kwargs["name"] = "GRNF"
        super().__init__(trainable=trainable, **kwargs)
        
        self.num_grnf = channels
        self.order_2_prc = order_2_prc
        
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
        
        self.mode = None
        self.mode_name = "not set"
    
    def build(self, input_shape):
        # inputs = [X, A [, E]]
        #   X:  (batch_size, N, F)   # F = num_node_features
        #   A:  (batch_size, N, N)
        #   E:  (n_edges, S) or (batch_size, N, N, S)   # S = num_edge_features
        
        if self.order_2_prc == 0:
            num_order1, num_order2 = self.num_grnf, 0
        elif self.order_2_prc == 1:
            num_order1, num_order2 = 0, self.num_grnf
        else:
            num_order2 = tf.cast(tf.reduce_sum(random_binomial(shape=(self.num_grnf,), p=self.order_2_prc)), tf.int32)
            num_order2 = max([min([self.num_grnf - 1, num_order2]), 1])
            num_order1 = self.num_grnf - num_order2
        assert (num_order1 + num_order2) == self.num_grnf
        
        self.mode, self.mode_name, self.input_pattern = get_spektral_mode(input_shape)
        self._pars["input_pattern"] = self.input_pattern

        print("Working in", self.mode_name)
        if self.mode not in [spektral_ops.modes.BATCH, SPEKTRAL_DISJOINT]:
            raise NotImplementedError("Mode {} not implemented.".format(self.mode_name))
        
        if self.mode == spektral_ops.modes.BATCH:
            self.psi_order_1 = NeuralFeatures1Batch(num_order1, name="NeuFeat1",
                                                    **self._pars) if num_order1 > 0 else None
            self.psi_order_2 = NeuralFeatures2Batch(num_order2, name="NeuFeat2",
                                                    **self._pars) if num_order2 > 0 else None
        if self.mode == SPEKTRAL_DISJOINT:
            self.psi_order_1 = NeuralFeatures1Disjoint(num_order1, name="NeuFeat1",
                                                       **self._pars) if num_order1 > 0 else None
            self.psi_order_2 = NeuralFeatures2Disjoint(num_order2, name="NeuFeat2",
                                                       **self._pars) if num_order2 > 0 else None
        
        self.concat = Concatenate(axis=-1) if num_order1 > 0 and num_order2 > 0 else None
    
    def call(self, inputs, **kwargs):
        if self.psi_order_1 is None:
            return self.psi_order_2(inputs)
        if self.psi_order_2 is None:
            return self.psi_order_1(inputs)
        psi1 = self.psi_order_1(inputs)
        psi2 = self.psi_order_2(inputs)
        psi = self.concat([psi1, psi2])
        # reweight to approximate distance and kernel
        if self._normalize:
            psi /= tf.math.sqrt(tf.cast(self.num_grnf, dtype=tf.float32))
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
        return self
