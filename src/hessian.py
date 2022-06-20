from src.subleading import *
import tensorflow as tf
import numpy as np

def make_eig_fnames(num_classes):
    h_evals_fname = 'h_evals_class_{}.npy'.format(num_classes)
    h_evecs_fname = 'h_evecs_class_{}.npy'.format(num_classes)
    return h_evals_fname, h_evecs_fname

def make_Hinv_from_eig(evals, evecs, data_length, num_classes):
    enum = len(evals)
    e_length = len(evecs[:, 0])

    evecs_rescaled = np.transpose([evalue ** (-1 / 2) * evecs[:, i] for i, evalue in enumerate(evals)])
    Hinv1 = evecs_rescaled @ np.transpose(evecs_rescaled)
    Hinv2 = Hinv1.reshape(data_length + 1, num_classes, data_length + 1, num_classes)
    return Hinv1, Hinv2


def make_hessian_train(model, X, Y, loss):
    n_model_variables = get_num_variables(model)
    num_classes = get_num_classes(model)

    with tf.GradientTape() as t2:
        with tf.GradientTape() as t1:
            y_true = Y.reshape(Y.shape[0], num_classes)
            y_pred = tf.reshape(model(X), (Y.shape[0], num_classes))
            loss_fun = loss(y_true, y_pred)
        g = t1.gradient(loss_fun, model.trainable_variables)
        g = tf.concat([tf.reshape(p, [-1, ]) for p in g], axis=0)

    hessian = t2.jacobian(g, model.trainable_variables)
    hessian = tf.concat([tf.reshape(p, [n_model_variables, -1]) for p in hessian], axis=1)

    return hessian