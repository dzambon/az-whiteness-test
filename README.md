# AZ-whiteness test

This repository is the official implementation of _AZ-whiteness test_ of our paper:   

Daniele Zambon, Cesare Alippi. [AZ-whiteness test: a test for signal uncorrelation on spatio-temporal graphs.](https://arxiv.org/abs/2204.11135) 2022.

```
@inproceedings{zambon2022aztest,
 author = {Zambon, Daniele and Alippi, Cesare},
 booktitle = {Advances in Neural Information Processing Systems},
 title={AZ-whiteness test: a test for signal uncorrelation on spatio-temporal graphs},
 year = {2022},
 url={https://arxiv.org/abs/2204.11135},
}
```

## Requirements

Experiments are run with Python3.8. All requirements should be specified in file `aztest-w-tsl.yml`. A working conda enviroment can be created by running
```bash
conda env create --file=aztest-w-tsl.yml
```

Datasets MetrLA and PemsBay are automatically retrieved from 
[TorchSpatiotemporal](https://github.com/TorchSpatiotemporal/tsl) library, 
whereas GPVAR data can be downloaded from the following link
[https://drive.switch.ch/index.php/s/qKaRyjJ0kSlnZN1](https://drive.switch.ch/index.php/s/qKaRyjJ0kSlnZN1)
and should be placed in folder `data/gpvar-T30000_line-c5`.


## Structure of the repository 

The structure of the repository is the following:

- `graph_sign_test.py` implements the AZ-test and some utilities to analyze the prediction residuals.
- `triangular_tricom_graph.py` implements a family of graphs, and exposes plotting utilities.
- `synthetic_residuals.py` implements several targeted synthetic tests on correlated graph signals.
- `graph_polynomial_var.py` implements the GPVAR model and provides a dataset class usable in [TorchSpatiotemporal](https://github.com/TorchSpatiotemporal/tsl) and torch nn model.
- `tsl_experiments.py` extends the [traffic forecasting example](https://github.com/TorchSpatiotemporal/tsl/blob/main/examples/prediction/run_traffic.py) in TorchSpatiotemporal adding GPVAR model and data.

Figures generated by script `synthetic_residuals.py` are stored in folder `results`, and results of `tsl_experiments.py` are stored in folder `log`.

Configuration files with model and training hyperparameters are in located in folder `config/traffic`.


## Experiments and figures of the paper

The experiments of the arXiv version of the paper are run with the following commands.

- Figure 3 is generated by calling `main("power-unimodal", disable_warning=True)` and `main("power-mixture", disable_warning=True)` in `synthetic_residual.py`.
- Table 2 (as well as its extension, Table 3 in supplementary material) is obtained from bash script `run_all_experiments.sh` that calls `tsl_experiments.py`; a summary table with all results is obtained by running `fetch_results.sh` after having listed the desired checkpoints from `./log` into it.
- Figure 4 and Figure 6 (supplementary material) are generated during the runs of `tsl_experiments.py` with `--dataset-name gpolyvar` and stored in folder `data/gpvar-T30000_line-c5`.  
- Figure 5 (supplementary material) is generated by calling `main("viz")` in `synthetic_residual.py`.
- Figure 7 (supplementary material) is generated by calling `main("sparse-full")` in `synthetic_residual.py`.
