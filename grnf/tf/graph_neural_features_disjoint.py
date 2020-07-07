import tensorflow as tf
from tensorflow.math import unsorted_segment_sum as tf_scatter_add
from .graph_neural_features_base import _NeuralFeaturesBase

def coalesce(indices, values):
    r""" It is basically scatter_add with bi-dimensional batch=indices """
    max_idx = tf.reduce_max(indices)
    tf.debugging.assert_less_equal(max_idx, tf.cast(2 ** 31, tf.int64), message="max num of nodes exceeded")
    dec = tf.ones(1, dtype=tf.int64)
    while dec <= max_idx:
        dec *= 2
    enc_index = indices[0] + dec * indices[1]
    coal_unique, coalesce_idx, _ = tf.unique_with_counts(enc_index)
    coal_values = tf_scatter_add(data=values, segment_ids=coalesce_idx, num_segments=tf.shape(coal_unique)[0])
    coal_index = tf.stack([coal_unique % dec, coal_unique // dec])
    return coal_index, coal_values

def get_normalization_factors(normalize, num_nodes):
    r""" Replicates the normalization factor to (batch_size, 1) (num_tot_nodes|batch, 1) """
    fact_b = tf.ones(1, dtype=tf.float32)
    fact_n = tf.ones(1, dtype=tf.float32)
    if normalize:
        fact_b = fact_b / tf.cast(num_nodes, tf.float32)
        fact_n = tf.expand_dims(tf.repeat(fact_b, num_nodes), 1)
        fact_b = tf.expand_dims(fact_b, 1)

    return fact_n, fact_b

class _NeuralFeaturesDisjoint(_NeuralFeaturesBase):
    r"""
    Extends the base class `grnf.tf._NeuralFeaturesBase` with the utilities for
    `disjoint` mode.
    """

    def parse_input(self, inputs):
        """
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

        The notation `num_tot_nodes|batch` stresses that they keep the batch subdivision of data.x,
        whereas `num_in_feat` is `max(1, num_node_features + num_edge_features)`.
        """

        if self.input_pattern == "XAEI":
            [X, A, E, I] = inputs
        elif self.input_pattern == "XAI":
            [X, A, I] = inputs
            E = None
        elif self.input_pattern == "AI":
            [A, I] = inputs
            E, X = None, None
        if E is None:
            E = tf.ones((tf.shape(A.indices)[0], 1), dtype=tf.float32)
        if X is None:
            X = tf.ones((tf.shape(I)[0], 0), dtype=tf.float32)
        
        # num_tot_nodes = data.batch.shape[0]  # total number of nodes
        num_tot_nodes = tf.shape(I)[0]  # total number of nodes
        batch_size = tf.reduce_max(I) + 1
        
        # Auxiliary vars
        diag_bidx = tf.where(A.indices[:, 0] == A.indices[:, 1])[:, 0] # nodes with self loops (diagonal)
    
        # normalization
        num_nodes = tf_scatter_add(data=tf.ones(tf.shape(I)[0], dtype=tf.int64), segment_ids=I, num_segments=batch_size)
        fact_n, fact_b = get_normalization_factors(self._normalize, num_nodes)

        # Compact representation
        #diag_part (num_tot_nodes|batch, num_in_feat)
        if len(diag_bidx) > 0:
            E_diag = tf.scatter_nd(indices=diag_bidx,
                                   updates=tf.gather(params=E, indices=diag_bidx),
                                   shape=tf.convert_to_tensor([num_tot_nodes, self.num_edge_features], dtype=tf.int64))
        else:
            E_diag = tf.zeros([num_tot_nodes, max([self.num_edge_features, 1])])
        diag_part = tf.concat([X, E_diag], axis=1)

        #sum_diag_part (batch_size, num_in_feat)
        sum_diag_part = tf_scatter_add(data=diag_part, segment_ids=I, num_segments=batch_size) * fact_b

        #sum_of_rows (num_tot_nodes|batch, num_in_feat)
        sum_of_rows = tf.concat([X, tf_scatter_add(data=E, segment_ids=A.indices[:, 0], num_segments=num_tot_nodes)], axis=1) * fact_n

        #sum_of_cols (num_tot_nodes|batch, num_in_feat)
        sum_of_cols = tf.concat([X, tf_scatter_add(data=E, segment_ids=A.indices[:, 1], num_segments=num_tot_nodes)], axis=1) * fact_n

        #sum_all = (batch_size, num_in_feat)
        sum_all = tf_scatter_add(data=sum_of_cols, segment_ids=I, num_segments=batch_size) * (fact_b**2)

        repr_compact = {
            "diag_part": diag_part,
            "sum_diag_part": sum_diag_part,
            "sum_of_rows": sum_of_rows,
            "sum_of_cols": sum_of_cols,
            "sum_all": sum_all,
        }
        
        return repr_compact, {"X": X, "A_indices": tf.transpose(A.indices), "E": E, "I": I}, num_nodes, fact_b

    def get_zerograph_representation(self, *args, **kwargs):
        """ Returns the embedding of the ``null'' graph to create `csi(g) = psi(g) - psi(0) """
        return self.call(inputs=None, zerograph=True, **kwargs)

    
class NeuralFeatures1Disjoint(_NeuralFeaturesDisjoint):
    r"""
    Extends `_NeuralFeaturesDisjoint` and implements GRNF all with hidden
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
        repr_eq = tf.einsum("nf, mhf -> nmh", repr_compact["diag_part"], self.kernel_equiv[:, 0, ...])
        # op2 - (123) + (12)(3) - tile sum of diag part
        repr_eq_sum = tf.einsum("bf, mhf -> bmh", repr_compact["sum_diag_part"], self.kernel_equiv[:, 1, ...])
        # op3 - (123) + (13)(2) - place sum of row i in element i
        repr_eq += tf.einsum("nf, mhf -> nmh", repr_compact["sum_of_rows"], self.kernel_equiv[:, 2, ...])
        # op4 - (123) + (23)(1) - place sum of col i in element i
        repr_eq += tf.einsum("nf, mhf -> nmh", repr_compact["sum_of_cols"], self.kernel_equiv[:, 3, ...])
        # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
        repr_eq_sum += tf.einsum("bf, mhf -> bmh", repr_compact["sum_all"], self.kernel_equiv[:, 4, ...])

        repr_eq += tf.repeat(repr_eq_sum, num_nodes, axis=0)

        #bias
        if self.use_bias:
            repr_eq += self.bias_equiv[..., 0, :]

        #activation
        repr_eq = self.hidden_activation(repr_eq)
        
        # --- Invariant part ----------------------------------

        #linear (with normalization)
        inv_basis = tf_scatter_add(data=repr_eq, segment_ids=data["I"], num_segments=tf.shape(repr_eq_sum)[0]) * fact_norm[..., None]
        repr_inv = tf.einsum("bmh, midh -> bmd", inv_basis, self.kernel_inv)
        
        #bias
        if self.use_bias:
            repr_inv += self.bias_inv
            
        return repr_inv[..., 0]
        

class NeuralFeatures2Disjoint(_NeuralFeaturesDisjoint):
    r"""
    Extends `_NeuralFeaturesDisjoint` and implements GRNF all with hidden
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

        #linear
        # op1 - (1234) - extract diag
        repr_eq_set_diag = tf.einsum("nf, mdf -> nmd", repr_compact["diag_part"], self.kernel_equiv[:, 0, ...])
        # op2 - (1234) + (12)(34) - place sum of diag on diag
        repr_eq_rep_diag = tf.einsum("bf, mdf -> bmd", repr_compact["sum_diag_part"], self.kernel_equiv[:, 1, ...])
        # op3 - (1234) + (123)(4) - place sum of row i on diag ii
        repr_eq_set_diag += tf.einsum("nf, mdf -> nmd", repr_compact["sum_of_rows"], self.kernel_equiv[:, 2, ...])
        # op4 - (1234) + (124)(3) - place sum of col i on diag ii
        repr_eq_set_diag += tf.einsum("nf, mdf -> nmd", repr_compact["sum_of_cols"], self.kernel_equiv[:, 3, ...])
        # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
        repr_eq_rep_diag += tf.einsum("bf, mdf -> bmd", repr_compact["sum_all"], self.kernel_equiv[:, 4, ...])
        # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
        repr_eq_rep_row = tf.einsum("nf, mdf -> nmd", repr_compact["sum_of_cols"], self.kernel_equiv[:, 5, ...])
        # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
        repr_eq_rep_row += tf.einsum("nf, mdf -> nmd", repr_compact["sum_of_rows"], self.kernel_equiv[:, 6, ...])
        # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
        repr_eq_rep_col = tf.einsum("nf, mdf -> nmd", repr_compact["sum_of_cols"], self.kernel_equiv[:, 7, ...])
        # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
        repr_eq_rep_col += tf.einsum("nf, mdf -> nmd", repr_compact["sum_of_rows"], self.kernel_equiv[:, 8, ...])
        # op10 - (1234) + (14)(23) - identity
        if self.num_node_features > 0:
            repr_eq_set_diag += tf.einsum("nf, mdf -> nmd", data["X"], self.kernel_equiv[:, 9, :, :self.num_node_features])
        repr_eq_E = tf.einsum("af, mdf -> amd", data["E"], self.kernel_equiv[:, 9, :, self.num_node_features:])
        # op11 - (1234) + (13)(24) - transpose
        if self.num_node_features > 0:
            repr_eq_set_diag += tf.einsum("nf, mdf -> nmd", data["X"], self.kernel_equiv[:, 10, :, :self.num_node_features])
        repr_eq_Et = tf.einsum("af, mdf -> amd", data["E"], self.kernel_equiv[:, 10, :, self.num_node_features:])
        # op12 - (1234) + (234)(1) - place ii element in row i
        repr_eq_rep_row += tf.einsum("nf, mdf -> nmd", repr_compact["diag_part"], self.kernel_equiv[:, 11, ...])
        # op13 - (1234) + (134)(2) - place ii element in col i
        repr_eq_rep_col += tf.einsum("nf, mdf -> nmd", repr_compact["diag_part"], self.kernel_equiv[:, 12, ...])
        # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
        repr_eq_rep_all = tf.einsum("bf, mdf -> bmd", repr_compact["sum_diag_part"], self.kernel_equiv[:, 13, ...])
        # op15 - sum of all ops - place sum of all entries in all entries
        repr_eq_rep_all += tf.einsum("bf, mdf -> bmd", repr_compact["sum_all"], self.kernel_equiv[:, 14, ...])

        repr_eq_set_diag += tf.repeat(repr_eq_rep_diag, num_nodes, axis=0)
        repr_eq_rep_col += tf.repeat(repr_eq_rep_all, num_nodes, axis=0)
        
        if self.use_bias:
            # bias can be directly addded here
            repr_eq_rep_col += self.bias_equiv[..., 1, :]
            repr_eq_set_diag += self.bias_equiv[..., 0, :] - self.bias_equiv[..., 1, :]

        # auxiliary variables
        n_, m_, b_, d_ = tf.shape(data["I"])[0], self.num_grnf, tf.shape(num_nodes)[0], self.num_hidden_features
        arange_ = tf.range(n_, dtype=tf.int64) # .unsqueeze_(0)
        num_nodes_num_nodes = tf.repeat(num_nodes, num_nodes)
        cc = tf.repeat(tf.cumsum(num_nodes), num_nodes)
        # tensors arange_repint_ and arange_repeat_ list the indices row-wise and column-wise
        arange_repint_ = tf.repeat(arange_, num_nodes_num_nodes)[None, ...]
        arange_repeat_ = tf.ragged.range(cc - num_nodes_num_nodes, cc).flat_values[None, ...]

        # this combines already the sum of rows and cols
        ct, ct2 = tf.cast(0, dtype=tf.int64), tf.cast(0, dtype=tf.int64)
        repr_eq_rep_full = tf.zeros((tf.shape(arange_repint_)[1], self.num_grnf, self.num_hidden_features), dtype=tf.float32)
        for b in range(0, len(num_nodes)):
            repr_eq_rep_full = tf.tensor_scatter_nd_update(repr_eq_rep_full,
                indices=tf.expand_dims(ct2 + tf.range(num_nodes[b]**2, dtype=tf.int64), 1),
                updates=tf.reshape(repr_eq_rep_row[ct: ct + num_nodes[b]][None, ...] \
                                 + repr_eq_rep_col[ct: ct + num_nodes[b]][:, None, ...],
                                   [-1, self.num_grnf, self.num_hidden_features]))
            ct += num_nodes[b]
            ct2 += num_nodes[b]**2

        # sparse tensors to prepare the repr_eq
        repr_eq_indices = tf.concat([
            tf.repeat(arange_[None, ...], 2, axis=0),            #diagonal
            tf.concat([arange_repint_, arange_repeat_], axis=0), #sum of rows and cols
            data["A_indices"],                                   #edge_attr
            data["A_indices"][::-1],                             #edge_attr transpose
        ], axis=1)

        repr_eq_values = tf.concat([
            repr_eq_set_diag,
            repr_eq_rep_full,
            repr_eq_E,
            repr_eq_Et
        ], axis=0)
        
        #sum everything
        repr_eq_indices, repr_eq_values = coalesce(repr_eq_indices, repr_eq_values)

        #activation
        repr_eq_values = self.hidden_activation(repr_eq_values)
        
        # --- Invariant part ----------------------------------
        
        #linear (with normalization)
        diag_idx = tf.where(repr_eq_indices[0] == repr_eq_indices[1])[:, 0]
        offd_idx = tf.where(repr_eq_indices[0] != repr_eq_indices[1])[:, 0]
        inv_basis_diag = tf_scatter_add(
            data=tf.gather(params=repr_eq_values, indices=diag_idx),
            segment_ids=data["I"],
            num_segments=b_) * tf.expand_dims(fact_norm, 1)
        inv_basis_offd = tf_scatter_add(
            data=tf.gather(params=repr_eq_values, indices=offd_idx),
            segment_ids=tf.gather(params=data["I"], indices=tf.gather(params=repr_eq_indices[0], indices=offd_idx)),
            num_segments=b_) * tf.expand_dims(fact_norm**2, 1)

        inv_basis = tf.stack([inv_basis_diag, inv_basis_offd], axis=3)
        repr_inv = tf.einsum("bmhi, midh -> bmd", inv_basis, self.kernel_inv)

        #bias
        if self.use_bias:
            repr_inv += self.bias_inv

        return repr_inv[..., 0]
        
