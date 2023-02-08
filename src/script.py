import pennylane as qml
import tensorflow as tf
import pandas as pd
import os
import json
import collections
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten
from sklearn.metrics import accuracy_score
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

n_layers = 4
tf.keras.backend.set_floatx('float64')

def filter_36(x, y):
    """Filter MNIST dataset to only labels 3 and 6"""
    keep = (y == 3) | (y == 6)
    x, y = x[keep], y[keep]
    y = y == 3
    return x,y

def remove_contradicting(xs, ys):
    mapping = collections.defaultdict(set)
    orig_x = {}
    # Determine the set of labels for each unique image:
    for x,y in zip(xs,ys):
       orig_x[tuple(x.flatten())] = x
       mapping[tuple(x.flatten())].add(y)
    
    new_x = []
    new_y = []
    for flatten_x in mapping:
      x = orig_x[flatten_x]
      labels = mapping[flatten_x]
      if len(labels) == 1:
          new_x.append(x)
          new_y.append(next(iter(labels)))
      else:
          # Throw out images that match more than one label.
          pass
    
    num_uniq_3 = sum(1 for value in mapping.values() if len(value) == 1 and True in value)
    num_uniq_6 = sum(1 for value in mapping.values() if len(value) == 1 and False in value)
    num_uniq_both = sum(1 for value in mapping.values() if len(value) == 2)
    
    return np.array(new_x), np.array(new_y)

def get_data(data_dir):
    """Read and transform data"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Rescale the images from [0,255] to the [0.0,1.0] range.
    x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0

    logger.info("Number of original training examples:" + str(len(x_train)))
    logger.info("Number of original test examples:" + str(len(x_test)))
    
    x_train, y_train = filter_36(x_train, y_train)
    x_test, y_test = filter_36(x_test, y_test)

    logger.info("Number of filtered training examples:" + str(len(x_train)))
    logger.info("Number of filtered test examples:" +  str(len(x_test)))
    
    x_train_small = tf.image.resize(x_train, (4,4)).numpy()
    x_test_small = tf.image.resize(x_test, (4,4)).numpy()
    x_train_nocon, y_train_nocon = remove_contradicting(x_train_small, y_train)
    x_test_nocon, y_test_nocon = remove_contradicting(x_test_small, y_test)
    
    return x_train_nocon, y_train_nocon, x_test_nocon, y_test_nocon

def main():
    logger.info("=== environment variables ==")
    logger.info(os.environ)
    
    input_dir = os.environ["AMZN_BRAKET_INPUT_DIR"]
    output_dir = os.environ["AMZN_BRAKET_JOB_RESULTS_DIR"]
    job_name = os.environ["AMZN_BRAKET_JOB_NAME"] 
    hp_file = os.environ["AMZN_BRAKET_HP_FILE"]
    device_arn = os.environ["AMZN_BRAKET_DEVICE_ARN"]
    
    logger.info("=== hyperparameters ===")
    with open(hp_file, "r") as f:
        hyperparams = json.load(f)
    logger.info(hyperparams)
    
    n_qubits = int(hyperparams["n_qubits"])
    n_shots = int(hyperparams["n_shots"])
    step = float(hyperparams["step"])
    batch_size = int(hyperparams["batch_size"])
    num_epochs = int(hyperparams["num_epochs"])
    max_parallel = int(hyperparams["max_parallel"])
    use_local_simulator = hyperparams["use_local_simulator"]
    
    if isinstance(use_local_simulator, str):
        use_local_simulator = (use_local_simulator == 'True')
    
    # Quantum Setup
    if use_local_simulator:
        dev = qml.device("default.qubit", wires=n_qubits)
        logger.info("Using local quantum simulator: default.qubit")
    else:
        dev = qml.device(
            "braket.aws.qubit",
            device_arn=device_arn,
            wires=n_qubits,
            shots=n_shots,
            s3_destination_folder=None,
            parallel=True,
            max_parallel=max_parallel,
        )
        logger.info(f"Using braket device: {device_arn}")
    
    data_dir=f"{input_dir}"
#     logger.info("=== data ===")
#     logger.info([os.path.join(dp, f) for dp, dn, fn in os.walk(os.path.expanduser(data_dir)) for f in fn])
# #     logger.info(os.listdir(data_dir))
    
    @qml.qnode(dev, interface="tf")
    def qnode(inputs, weights):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.BasicEntanglerLayers(weights, wires=range(n_qubits))
        return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

    X_train, y_train, X_test, y_test = get_data(data_dir)
    logger.info(X_train.shape)
    logger.info(y_train.shape)
    
    n_features = X_train.shape[1]
    logger.info("n_features: " + str(n_features))
    weight_shapes = {"weights": (n_layers, n_qubits)}
    model = Sequential()
    model.add(Flatten(input_shape=(n_features, n_features)))
    model.add(Dense(4, activation='relu'))
    model.add(qml.qnn.KerasLayer(qnode, weight_shapes, output_dim=n_qubits))
    model.add(Dense(8, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1, activation="sigmoid"))
        
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=["accuracy"])
    model.fit(X_train, y_train, epochs=num_epochs, batch_size=batch_size, verbose=1)
    model_dir = os.path.join(output_dir, "v1.h5")
    model.save(model_dir)
    
    # Inference - Requires h5py library
    # model = tf.keras.models.load_model(model_dir)
    # predictions = model.predict(X_test)
    # test_accuracy = accuracy_score(y_test, predictions)
    # logger.info(test_accuracy)
    
if __name__ == "__main__":
    main()
    
    

    