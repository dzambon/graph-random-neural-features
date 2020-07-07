from grnf import utils

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras import activations, initializers
from spektral.layers import ops as spektral_ops

class _NeuralFeaturesBase(Layer):
    r"""
    Base class for graph neural features of a predefined hidden tensor order
    ([Zambon et al. 2020]([https://arxiv.org/abs/1909.03790)).
    """
    
    def __init__(self,
                 channels, # out_feature
                 hidden_features=None,
                 activation=None,
                 hidden_activation="relu",
                 hidden_tensor_order=1,
                 use_bias=True,
                 kernel_initializer=tf.random_normal_initializer(mean=0., stddev=1.),
                 bias_initializer=tf.random_normal_initializer(mean=0., stddev=1.),
                 normalize=True,
                 input_pattern=None,
                 center_embedding=True,
                 **kwargs):

        super().__init__(**kwargs)


        if activation is None:
            activation = "linear"
        if hidden_activation is None:
            hidden_activation = "linear"
        self.activation = activations.get(activation)
        self.hidden_activation = activations.get(hidden_activation)
        self.use_bias = use_bias
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)
        
        self.num_hidden_features = hidden_features
        self.num_grnf = channels
        self._normalize = normalize
        self.hidden_tensor_order = hidden_tensor_order
        self.input_pattern = input_pattern
        
        self.center_embedding = center_embedding
    
    def build(self, input_shape):
        # inputs = [X, A, E]]
        #   X:  (batch_size, N, F)   # F = num_node_features
        #   A:  (batch_size, N, N)
        #   E:  (n_edges, S) or (batch_size, N, N, S)   # S = num_edge_features
        Xi = self.input_pattern.find('X')
        self.num_node_features = input_shape[Xi][-1] if Xi >= 0 else 0
        Ei = self.input_pattern.find('E')
        self.num_edge_features = input_shape[Ei][-1] if Ei >= 0 else 0

        if self.num_hidden_features is None:
            self.num_hidden_features = max([1, 2 * (self.num_node_features + self.num_edge_features)])

        self.reset_parameters()

        # precompute centering
        if self.center_embedding:
            self.zerograph = self.activation(self.compute_zerograph())

    def reset_parameters(self):

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

        self.kernel_equiv = self.add_weight(shape=eq_k_sh,
                                            initializer=self.kernel_initializer,
                                            name='kernel_equiv')
        if self.use_bias:
            self.bias_equiv = self.add_weight(shape=eq_b_sh,
                                              initializer=self.bias_initializer,
                                              name='bias_equiv')

        self.kernel_inv = self.add_weight(shape=in_k_sh,
                                            initializer=self.kernel_initializer,
                                            name='kernel_inv')
        if self.use_bias:
            self.bias_inv = self.add_weight(shape=in_b_sh,
                                            initializer=self.bias_initializer,
                                            name='bias_inv')

    @property
    def num_in_features(self):
        """ When there are no edge features, then the adjacency matrix is considered. """
        return self.num_node_features + max([1, self.num_edge_features])

    def call(self, inputs, zerograph=False, **kwargs):
        # first convenient input representation, a list with shapes
        #     - (batch_size, num_nodes, num_in_feat) or (num_tot_nodes|batch, num_in_feat)
        #     - (batch_size,         1, num_in_feat).
        repr_compact, data, num_nodes, fact_norm = self.parse_input(inputs=inputs)
        
        # compute neural features psi
        psi = self.compute_neural_features(repr_compact=repr_compact, data=data,
                                           num_nodes=num_nodes, fact_norm=fact_norm)
        psi = self.activation(psi)

        if self.center_embedding:
            psi -= self.zerograph

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
            repr_inv = tf.einsum("bmh, mdh -> bmd", inv_basis, self.kernel_inv[:, 0, ...])
            repr_inv += self.bias_inv
        else:
            repr_inv = tf.zeros((1, self.num_grnf, 1))
        return  repr_inv[..., 0]

    def get_grnf_weights(self):
        w = {}
        w["kernel_equiv"] = self.kernel_equiv.numpy()
        w["kernel_inv"] = self.kernel_inv.numpy()
        if self.use_bias:
            w["bias_equiv"] = self.bias_equiv.numpy()
            w["bias_inv"] = self.bias_inv.numpy()
        return w
    
    def set_grnf_weights(self, w):
        assert self.kernel_equiv.shape == w["kernel_equiv"].shape
        assert self.kernel_inv.shape == w["kernel_inv"].shape
        self.kernel_equiv.assign(tf.convert_to_tensor(w["kernel_equiv"]))
        self.kernel_inv.assign(tf.convert_to_tensor(w["kernel_inv"]))
        self.num_grnf = w["kernel_inv"].shape[0]

        if self.use_bias:
            assert self.bias_equiv.shape == w["bias_equiv"].shape
            assert self.bias_inv.shape == w["bias_inv"].shape
            self.bias_equiv.assign(tf.convert_to_tensor(w["bias_equiv"]))
            self.bias_inv.assign(tf.convert_to_tensor(w["bias_inv"]))

        return self
