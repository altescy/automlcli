AutoMLCLI
=========

[![Actions Status](https://github.com/altescy/automlcli/workflows/CI/badge.svg)](https://github.com/altescy/automlcli/actions?query=workflow%3ACI)

AutoMLCLI is a simple AutoML command line tool for tabular data.


### Features
- Enables you to train and evaluate model / make prediction from command line without writing python scripts
- Read / write data from web or cloud strages
- Mange experimental results with [MLflow](https://github.com/mlflow/mlflow)
- Highly extensible by plugin system

### Supported models
- [TPOT](https://github.com/EpistasisLab/tpot)
- [FLAML](https://github.com/microsoft/FLAML)


## Installation
```
$ pip install git+https://github.com/altescy/automlcli.git
$ pip install "automlcli[all]"
```

## Usage

#### Train a model
```
$ automl train config.yml train.csv --serialization-dir out
$ ls out
best.json  flaml.log  metrics.json  model.pkl
```

#### Evaluate the trained model
```
$ automl evaluate out/model.pkl dev.csv --cv 5 --scoring accuracy --scoring f1_macro
```

#### Make prediction
```
$ automl predict out/model.pkl test.csv --output-file predictions.pkl
```

