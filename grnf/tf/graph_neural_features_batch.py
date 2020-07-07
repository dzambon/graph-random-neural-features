import tensorflow as tf
from tensorflow.math import unsorted_segment_sum as tf_scatter_add
from .graph_neural_features_base import _NeuralFeaturesBase

def set_middle_diag(T):
    r""" Set the values in T (a, b, c) to the middle diagonal (a, b, b, c) """
    naxis = len(tf.shape(T)) + 1
    p1 = list(range(naxis))
    p1[-1] = p1[-3]
    p1[-3] = p1[-2]
    p1[-2] = naxis - 1

    p2 = list(range(naxis-1))
    p2[-1] = p2[-2]
    p2[-2] = naxis - 2

    res = tf.transpose(tf.linalg.diag(tf.transpose(T, perm=p2)), perm=p1)
    return res

def get_middle_diag(T):
    r""" Get the middle diagonal (a, b, c) values in T (a, b, b, c) """
    naxis = len(tf.shape(T))
    p1 = list(range(naxis))
    p1[-1] = p1[-2]
    p1[-2] = p1[-3]
    p1[-3] = naxis - 1

    p2 = list(range(naxis-1))
    p2[-1] = p2[-2]
    p2[-2] = naxis - 2

    res = tf.transpose(tf.linalg.diag_part(
                            tf.transpose(T, perm=p1)), perm=p2)
    return res


class _NeuralFeaturesBatch(_NeuralFeaturesBase):
    r"""
    Extends the base class `grnf.tf._NeuralFeaturesBase` with the utilities for
    `batch` mode.
    """

    def parse_input(self, inputs):
        """
        Create a first compact and convenient representation of the graphs; see also Appendix A
        in [Maron et al. 2019](https://arxiv.org/abs/1812.09902).

        The output representation is a list containing
        ```
            repr_a = [
                diag_part:     (batch_size, num_nodes, num_in_feat)
                sum_diag_part: (batch_size,         1, num_in_feat)
                sum_of_rows:   (batch_size, num_nodes, num_in_feat)
                sum_of_cols:   (batch_size, num_nodes, num_in_feat)
                sum_all:       (batch_size,         1, num_in_feat)
            ]
        ```

        The notation `num_tot_nodes|batch` stresses that they keep the batch subdivision of data.x,
        whereas `num_in_feat` is `max(1, num_node_features + num_edge_features)`.
        """
        if self.input_pattern == "XAE":
            [X, A, E] = inputs
        elif self.input_pattern == "XA":
            [X, A] = inputs
            E = tf.expand_dims(A, axis=-1)
            tf.dtypes.cast(E, tf.float32)
        elif self.input_pattern == "A":
            [A] = inputs
            X = tf.zeros((tf.shape(A)[0], tf.shape(A)[1], 0))
            E = tf.expand_dims(A, axis=-1)
            tf.dtypes.cast(E, tf.float32)
        else:
            raise NotImplementedError
        assert A.shape[1] == A.shape[2] and E.shape[1] == E.shape[2]

        # normalization factor
        num_nodes = tf.shape(A)[1]
        fact = 1.
        if self._normalize:
            fact = tf.divide(tf.ones(1, dtype=tf.float32), tf.cast(num_nodes, tf.float32))
    
        repr_compact = {}
        
        # diag_part = tf.matrix_diag_part(inputs)   # N x D x m
        # (b, n, f) <- (b, n, f), (b, n, n, f)
        repr_compact["diag_part"] = tf.concat([X, get_middle_diag(E)], axis=2)
        # sum_diag_part = tf.reduce_sum(diag_part, axis=2, keepdims=True)  # N x D x 1
        # (b, 1, f) <- (b, n, f)
        repr_compact["sum_diag_part"] = tf.reduce_sum(repr_compact["diag_part"], axis=1, keepdims=True) * fact
        # sum_of_rows = tf.reduce_sum(inputs, axis=3)  # N x D x m
        # (b, n, f) <- (b, n, f), (b, n, n, f)
        repr_compact["sum_of_rows"] = tf.concat([X, tf.reduce_sum(E, axis=2)], axis=2) * fact
        # sum_of_cols = tf.reduce_sum(inputs, axis=2)  # N x D x m
        # (b, n, f) <- (b, n, f), (b, n, n, f)
        repr_compact["sum_of_cols"] = tf.concat([X, tf.reduce_sum(E, axis=1)], axis=2) * fact
        # sum_all = tf.reduce_sum(sum_of_rows, axis=2)  # N x D
        # (b, f) <- (b, n, f), (b, n, n, f)
        repr_compact["sum_all"] = tf.reduce_sum(repr_compact["sum_of_cols"], axis=1, keepdims=True) * fact ** 2
    
        data = {
            "X": X,
            "A": A,
            "E": E,
        }
        return repr_compact, data, num_nodes, fact
    

class NeuralFeatures1Batch(_NeuralFeaturesBatch):
    r"""
    Extends `_NeuralFeaturesBatch` and implements GRNF all with hidden
    tensor order equal to 1. 
    """

    def __init__(self, channels, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "NeuFeat1"
        assert kwargs.pop("hidden_tensor_order", 1) == 1
        super().__init__(channels=channels, hidden_tensor_order=1, **kwargs)
        assert self.hidden_tensor_order == 1

    def compute_neural_features(self, repr_compact, data, num_nodes, fact_norm):
        r"""
        The input data comes in the `repr_compact`, that is
        ```
            repr_compact = [diag_part, sum_diag_part, sum_of_rows, sum_of_cols, sum_all],
        which have shapes (batch_size, num_nodes, num_in_feat) and (batch_size, 1, num_in_feat).

        The equivariant representation is constructed from compact one, as a linear 
        combination of Bell(k_hid+2) components, resulting in `repr_eq` with shape
        `(batch_size, num_nodes, num_neural_feat, num_hidden_feat)`.

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
        # stack into a single tensor
        repr_tensor = tf.stack([
            # op1 - (123) - extract diag
            repr_compact["diag_part"],
            # op2 - (123) + (12)(3) - tile sum of diag part
            tf.tile(repr_compact["sum_diag_part"], [1, num_nodes, 1]),
            # op3 - (123) + (13)(2) - place sum of row i in element i
            repr_compact["sum_of_rows"],
            # op4 - (123) + (23)(1) - place sum of col i in element i
            repr_compact["sum_of_cols"],
            # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
            tf.tile(repr_compact["sum_all"], [1, num_nodes, 1])
        ])
        
        # --- Equivariant part  -----------------------------

        #linear
        repr_eq = tf.einsum("ebnf, mehf -> bmnh", repr_tensor, self.kernel_equiv)  # (Rand_feat, N, ..., N, Feat_hidden)

        #bias
        if self.use_bias:
            repr_eq += self.bias_equiv

        #activation
        repr_eq = self.hidden_activation(repr_eq)
        
        # --- Invariant part ----------------------------------

        #linear (with normalization)
        inv_basis = tf.reduce_sum(repr_eq, axis=2, keepdims=True) * fact_norm # (Batch, Num_rand_feat, Gamma_i=1, Feat_hid)
        repr_inv = tf.einsum("bmif, midf -> bmd", inv_basis, self.kernel_inv)  # (Num_rand_feat, Feat_inv=1)
        
        #bias
        if self.use_bias:
            repr_inv += self.bias_inv
            
        return repr_inv[..., 0]

class NeuralFeatures2Batch(_NeuralFeaturesBatch):
    r"""
    Extends `_NeuralFeaturesBatch` and implements GRNF all with hidden
    tensor order equal to 2. 
    """

    def __init__(self, channels, **kwargs):
        if "name" not in kwargs:
            kwargs["name"] = "NeuFeat2"
        assert kwargs.pop("hidden_tensor_order", 2) == 2
        super().__init__(channels=channels, hidden_tensor_order=2, **kwargs)
        assert self.hidden_tensor_order == 2

    def compute_neural_features(self, repr_compact, data, num_nodes, fact_norm):
        """
        The input data comes in the `repr_compact`, that is
        ```
            repr_compact = [diag_part, sum_diag_part, sum_of_rows, sum_of_cols, sum_all],
        ```
        which have shapes (batch_size, num_nodes, num_in_feat) and (batch_size, 1, num_in_feat).

        The equivariant representation is constructed from compact one, as a linear
        combination of Bell(k_hid+2) components.

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

        # stack into a single tensor
        repr_tensor = tf.stack([
            # op1 - (1234) - extract diag
            set_middle_diag(repr_compact["diag_part"]),
            # op2 - (1234) + (12)(34) - place sum of diag on diag
            set_middle_diag(tf.tile(repr_compact["sum_diag_part"], [1, num_nodes, 1])),
            # op3 - (1234) + (123)(4) - place sum of row i on diag ii
            set_middle_diag(repr_compact["sum_of_rows"]),
            # op4 - (1234) + (124)(3) - place sum of col i on diag ii
            set_middle_diag(repr_compact["sum_of_cols"]),
            # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
            set_middle_diag(tf.tile(repr_compact["sum_all"], [1, num_nodes, 1])),
            # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
            tf.tile(tf.expand_dims(repr_compact["sum_of_cols"], 2), [1, 1, num_nodes, 1]),
            # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
            tf.tile(tf.expand_dims(repr_compact["sum_of_rows"], 2), [1, 1, num_nodes, 1]),
            # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
            tf.tile(tf.expand_dims(repr_compact["sum_of_cols"], 1), [1, num_nodes, 1, 1]),
            # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
            tf.tile(tf.expand_dims(repr_compact["sum_of_rows"], 1), [1, num_nodes, 1, 1]),
            # op10 - (1234) + (14)(23) - identity
            tf.concat([set_middle_diag(data["X"]), data["E"]], axis=3),
            # op11 - (1234) + (13)(24) - transpose
            tf.transpose(tf.concat([set_middle_diag(data["X"]), data["E"]], axis=3), perm=[0, 2, 1, 3]),
            # op12 - (1234) + (234)(1) - place ii element in row i
            tf.tile(tf.expand_dims(repr_compact["diag_part"], axis=2), [1, 1, num_nodes, 1]),
            # op13 - (1234) + (134)(2) - place ii element in col i
            tf.tile(tf.expand_dims(repr_compact["diag_part"], axis=1), [1, num_nodes, 1, 1]),
            # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
            tf.tile(tf.expand_dims(repr_compact["sum_diag_part"], axis=1), [1, num_nodes, num_nodes, 1]),
            # op15 - sum of all ops - place sum of all entries in all entries
            tf.tile(tf.expand_dims(repr_compact["sum_all"], axis=1), [1, num_nodes, num_nodes, 1]),
        ])
       
        # --- Equivariant part  -----------------------------
        
        #linear
        repr_eq = tf.einsum("ebnlf, medf -> bmnld", repr_tensor, self.kernel_equiv)  # (Rand_feat, N, ..., N, Feat_hidden)

        #bias
        if self.use_bias:
            # mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.eye(tf.shape(repr_eq)[2]), 0) ,0), -1)
            mask = tf.eye(tf.shape(repr_eq)[2])[None, None, ..., None]
            repr_eq += tf.multiply(mask, tf.expand_dims(self.bias_equiv, 3)[:, :, :1, ...])
            repr_eq += tf.multiply(1.-mask, tf.expand_dims(self.bias_equiv, 3)[:, :, 1:, ...])
    

        #activation
        repr_eq = self.hidden_activation(repr_eq)
        
        # --- Invariant part ----------------------------------

        #linear (with normalization)
        tmp = tf.reduce_sum(get_middle_diag(repr_eq), axis=2)
        inv_basis = tf.stack([
            tmp * fact_norm,
            (tf.reduce_sum(repr_eq, axis=[2, 3]) - tmp) * fact_norm**2 ,
        ], axis=3) # (Batch, Num_rand_feat, Gamma_i=2, Feat_out)
        repr_inv = tf.einsum("bmhi, midh -> bmd", inv_basis, self.kernel_inv)

        #bias
        if self.use_bias:
            repr_inv += self.bias_inv
    
        return repr_inv[..., 0]

