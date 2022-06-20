import pandas as pd
pd.set_option('display.max_rows', 500)

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import initializers
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

from src.mnist import *

MNIST_path = os.path.join(os.path.expanduser('~'), '.keras/datasets/mnist.npz')
ROOT_DIR = '/Users/halmagyi/Documents/MachineLearning/ML_Notes/BaysianNNets/BayesianNets'
# ROOT_DIR = '/home/ubuntu/Documents/BayesianNets'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
os.chdir(ROOT_DIR)

def make_model_fname(num_classes, num_layers, width):
    if num_layers >0:
        model_fname = "model_class_{}_layers_{}_width_{}.model".format(num_classes, num_layers, width)
    else:
        model_fname = "model_class_{}_layers_{}.model".format(num_classes, num_layers)
    return model_fname


def make_mnist_model(num_classes, num_layers, data_length, hidden_width, seed=42, output_l2=0.001):

    hidden_activation = 'relu'
    output_activation = 'softmax'

    stddev = data_length ** (-1 / 2)
    kernel_initializer = initializers.RandomNormal(mean=0.0, stddev=stddev, seed=seed)

    use_bias = True
    dropout = False


    ################
    # Regularizer
    ################
    hidden_l2 = 0.00
    hidden_regularizer = tf.keras.regularizers.l2(hidden_l2)

    output_kernel_regularizer = tf.keras.regularizers.l2(output_l2)

    ################
    # Model
    ################

    inputs = Input(shape=(data_length,))

    if num_layers > 0:
        x = Dense(hidden_width, activation=hidden_activation,
                  use_bias=use_bias,
                  kernel_initializer=kernel_initializer,
                  kernel_regularizer=hidden_regularizer)(inputs)
        if dropout:
            x = Dropout(0.2)(x)

        for i in range(num_layers - 1):
            x = Dense(hidden_width, activation=hidden_activation,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=hidden_regularizer)(x)
            if dropout:
                x = Dropout(0.2)(x)
    else:
        x = inputs

    outputs = Dense(num_classes,
                    activation=output_activation,
                    kernel_initializer=kernel_initializer,
                    use_bias=use_bias,
                    kernel_regularizer=output_kernel_regularizer)(x)

    # This creates a model that includes
    # the Input layer and three Dense layers
    model = Model(inputs=inputs, outputs=outputs)


    return model
