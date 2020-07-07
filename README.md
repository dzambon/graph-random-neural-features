# Graph Random Neural Features

Graph Random Neural Features (GRNF) is an embedding method from graph-structured data to real vectors based on a family of graph neural networks. 
GRNF can be used within traditional processing methods or as a training-free input layer of a graph neural network. 
The theoretical guarantees that accompany GRNF ensure that the considered graph distance is metric, hence allowing to distinguish any pair of non-isomorphic graphs, and that GRNF approximately preserves its metric structure. 

The following is the reference paper:

#### [Graph Random Neural Features for Distance-Preserving Graph Representations](https://arxiv.org/abs/1909.03790).

```
@inproceedings{zambon2020graph,
  title={Graph Random Neural Features for Distance-Preserving Graph Representations},
  author={Daniele Zambon, Cesare Alippi and Lorenzo Livi},
  year={2020},
  booktitle={To appear in International Conference on Machine Learning (ICML) 2020},
}
```


## Paper experiments


### Convergence increasing the embedding dimension

```bash
python verify_bounds.py
python accuracy_convergence.py del
python accuracy_convergence.py sbm
```


### GRNF as a layer of a neural net

Get into the folder `gnn-comparison`. The framework has been obtained from [gnn-comparison](https://github.com/diningphil/gnn-comparison) (commit `6c2547b`) and contains some additional code.
From the script `red_button_202001.sh` you should be able to replicate all the experiments.


## ChangeLog

### v-0.1.0
* Experiments of the paper
