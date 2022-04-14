# DAPDAG
Master Thesis Project

This repository contains the PyTorch source codes to reproduce part of the simulation experiments in my Master Thesis Project.

## Main Codes
- **DAPDAGBayes.py**: main script for model architecture.
- **SimulateData.R**: R program for producing synthetic regression and classification datasets.
- **synthetic_train.py**: main script for training model on selected synthetic datasets.

## Supporting Codes

./utils: components/modules required for building the model and training.

## Miscellaneous

### Introduction of Model Architecture
The model structure is depicted as in ![model](./figures/dapcastion.pdf).

### True Causal Graphs of Synthetic Datasets
The synthetic regression and classification datasets are generated according to the causal graphs in ![graph1](./figures/regdag2.pdf) and ![graph2](./figures/cla_dag.pdf).
