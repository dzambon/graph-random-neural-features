In order to run the GRNF in the [gnn-comparison](https://github.com/diningphil/gnn-comparison) framework:

* Download the code from the [GitHub repository](https://github.com/diningphil/gnn-comparison).
* Copy the configuration files `config_GRNF.yml` and `config_GRNF_cpu.yml` in `gnn-comparison/` folder.
* Copy the model file `GRNF.py` to `gnn-comparison/models/graph_classifiers/` folder.
* Add the new model in file `config/base.py`:
    - Import the model `from models.graph_classifiers.GRNF import GRNF`.
    - Add it to `Config.models` (around line 42) `"GRNF": GRNF`