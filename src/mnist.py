import numpy as np
import os
from tensorflow.keras.utils import to_categorical

MNIST_path = os.path.join(os.path.expanduser('~'), '.keras/datasets/mnist.npz')


# tf.keras.datasets.mnist.load_data(path="mnist.npz")

def make_mnist_data(num_classes):
    data = np.load(MNIST_path)

    # MNIST arrives already split into train/test,
    x_train = data['x_train']/255.0
    x_test = data['x_test']/255.0
    y_train = data['y_train']
    y_test = data['y_test']


    y_train_cat = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

    y01_train = np.array(list(map(int,[y < 5 for y in y_train]))).reshape(-1,1)
    y01_test = np.array(list(map(int,[y < 5 for y in y_test]))).reshape(-1,1)

    y012_train = to_categorical(np.array(list(map(int,[y/4 for y in y_train]))))
    y012_test = to_categorical(np.array(list(map(int,[y/4 for y in y_test]))))

    y01234_train = to_categorical(np.array(list(map(int,[y/2 for y in y_train]))))
    y01234_test = to_categorical(np.array(list(map(int,[y/2 for y in y_test]))))


    x_train_flat = np.array([img.flatten() for img in x_train])
    x_test_flat = np.array([img.flatten() for img in x_test])



    # add unit for bias
    x_train_flat_bias = [np.concatenate([x, [1]], axis=0) for x in x_train_flat]
    x_test_flat_bias = [np.concatenate([x, [1]]) for x in x_test_flat]

    # train_size = len(x_train_flat)
    if num_classes == 10:
        Y_train = y_train_cat
        Y_test = y_test_cat
    elif num_classes == 5:
        Y_train = y01234_train
        Y_test = y01234_test
    elif num_classes == 3:
        Y_train = y012_train
        Y_test = y012_test
    elif num_classes == 1:
        loss = tf.keras.losses.BinaryCrossentropy()
        output_activation = 'sigmoid'
        Y_train = y01_train
        Y_test = y01_test

    return x_train_flat, x_test_flat, x_train_flat_bias, x_test_flat_bias, Y_train, Y_test