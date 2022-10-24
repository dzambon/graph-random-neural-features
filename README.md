# Graph Random Neural Features

Graph Random Neural Features (GRNF) is an embedding method from graph-structured data to real vectors based on a family of graph neural networks. 
GRNF can be used within traditional processing methods or as a training-free input layer of a graph neural network. 
The theoretical guarantees that accompany GRNF ensure that the considered graph distance is metric, hence allowing to distinguish any pair of non-isomorphic graphs, and that GRNF approximately preserves its metric structure. 

The following is the reference paper:

#### [Graph Random Neural Features for Distance-Preserving Graph Representations](https://proceedings.mlr.press/v119/zambon20a.html).

```
@inproceedings{zambon2020graph,
  title={Graph Random Neural Features for Distance-Preserving Graph Representations},
  author={Zambon, Daniele and Alippi, Cesare and Livi, Lorenzo},
  booktitle={Proceedings of the 37th International Conference on Machine Learning (ICML)},
  pages={10968--10977},
  editor={Hal Daumé III and Aarti Singh},
  volume={119},
  series={Proceedings of Machine Learning Research},
  address={Virtual},
  year={2020},
  month={13--18 Jul},
  publisher={PMLR},
  pdf={http://proceedings.mlr.press/v119/zambon20a/zambon20a.pdf},
}
```


## Implementation

GRNF has both a [Spektral](graphneural.network) (TensorFlow) implementation:

```python
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from grnf.tf import GraphRandomNeuralFeatures
X_in = Input(shape=(N, F))
A_in = Input(shape=(N, N))
psi = GraphRandomNeuralFeatures(64, activation="relu")([X_in, A_in])
output = Dense(1)(psi)
model = Model(inputs=[X_in, A_in], outputs=output)
```

as well as a [PyTorch Geometric](https://github.com/rusty1s/pytorch_geometric) one:
```python
from grnf.torch import GraphRandomNeuralFeatures
grnf = GraphRandomNeuralFeatures(64)
z = grnf(data)
```


## Examples

Adapted from [Spektral examples](https://graphneural.network/examples/):

* `examples/tf/delaunay_batch.py`: Graph classification 
* `examples/tf/qm9_batch.py`: Graph regression 
* `examples/tf/qm9_disjoint.py`: Graph regression 

Moreover, in `examples/torch` you can find also the code to test GRNF in the [gnn-comparison](https://github.com/diningphil/gnn-comparison) framework.


## Paper experiments

Please, refer to branch `v0.1.0`
```bash
git checkout v0.1.0
```


## ChangeLog

### v0.2.0
* Improved `pytorch` implementation
* Added Spektral implementation working in `batch` and `disjoint` modes.
* Added examples

### v0.1.0
* Experiments of the paper
