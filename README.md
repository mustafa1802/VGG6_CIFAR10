# VGG6_CIFAR10
This repository contains results from the experiments conducted as part of an assignment of CS6886W - Systems Engineering for Deep Learning. This assignment focuses on exploring VGG6 on CIFAR-10 with different configurations  and analyzing model performance based on these configurations.

# Task
To run W&B sweeps across various hyperparameters of the VGG6 model on the CIFAR-10 dataset and choose the best performing model (based on the W&B parallel plot). Using the confguration of the best performing model, train the VGG6 model using the CIFAR-10 dataset and report the top-1 test accuracy for the best performing model

# Repoitory Structure

- model/ - contains model script with requirements for running on the local machine
- collab notebooks/ - contains ipynb files for reference

# Steps to reproduce the best model performance

## Setting up conda (optional)
Refer to the official documentation for more details - https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html

## Installing dependencies
```pip install -r requirements.txt```

## Running the VGG6 model
`python dl_assignment_1_vgg6_model.py`

**Python 3.13.5** was used while running the scripts locally
