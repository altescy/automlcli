AutoMLCLI
=========

[![Actions Status](https://github.com/altescy/automlcli/workflows/CI/badge.svg)](https://github.com/altescy/automlcli/actions?query=workflow%3ACI)


```
# Install AutoMLCLI
$ pip install git+https://github.com/altescy/automlcli.git
$ pip install automlcli[all]

# Train model
$ automl train config.yml train.csv model.pkl

# Make prediction
$ automl predict model.pkl test.csv --output-file preds.pkl
```
