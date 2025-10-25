# VGG6_CIFAR10
This repository contains results from the experiments conducted as part of an assignment of CS6886W - Systems Engineering for Deep Learning. This assignment focuses on exploring VGG6 on CIFAR-10 with different configurations  and analyzing model performance based on these configurations.

# Task
To run W&B sweeps across various hyperparameters of the VGG6 model on the CIFAR-10 dataset and choose the best performing model (based on the W&B parallel plot). Using the confguration of the best performing model, train the VGG6 model using the CIFAR-10 dataset and report the top-1 test accuracy for the best performing model

# Top Model Configuration
```
-----------------------------
Hyperparameter	      Value |
----------------------------|
Activation Function	|  GELU |
Optimizer	        |  Adam |
Learning Rate	    |  0.001|
Batch Size	        |  32   |
Epochs	            |  50   |
Batch_norm	        |  True |
-----------------------------
```

# Repository Structure

- model/ - contains model script (training script), test script, trained model and the requirements.txt file to reproduce test results
- collab notebooks/ - contains ipynb files for reference (used for W&B sweeps and the model training)

# Steps to reproduce the best model performance
All relevant scripts and files are present in the `model/` folder of this repository

Follow the below steps to setup your local system to run the model scripts

## Setting up conda (optional)
Refer to the official documentation for more details - https://docs.conda.io/projects/conda/en/stable/user-guide/getting-started.html

## Installing dependencies
```pip install -r requirements.txt```

## Running the VGG6 model
`python dl_assignment_1_vgg6_model.py`

## Running the eval script using the trained VGG6 model
`python test.py`

# System Configuration
```
Package                   Installed Version
---------------------------------------------
torch                     2.9.0
numpy                     2.3.4
Pillow                    12.0.0
matplotlib                3.10.7
torchvision               0.24.0
Python                    3.13.5
