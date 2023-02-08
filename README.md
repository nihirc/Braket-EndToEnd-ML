# Overview
The demo shows how to build end to end pipeline for QML. The demo uses Pennylane with TensorFlow interface. The demo builds binary classifier using MNIST dataset and uses Tensorflow interface with Pennylane to build hybrid model. 

# How to run
* Update hyperparameters in `run.py` file.
* Verify region on `line 8` and update bucket on `line 13`. 

## How to run locally
* The code will download Pennylane Tensorflow interface container image provided by AWS.
* Once the image is downloaded, line 27 to 35 will start running QML job locally.

## How to run as a job on Amazon Braket
* Set hyperparameter `use_local_simulator` to False to run as a job on Amazon Braket

## How to run on quantum hardware
* Set `q_device` variable to quantum hardware ARN to run on quantum hardware
* Ensure that `use_local_simulator` is set to False