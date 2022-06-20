import numpy as np
import tensorflow as tf
import time

from sklearn.metrics import accuracy_score as acc

def delta(x, y):
    if x == y:
        return 1
    else:
        return 0

def get_num_variables(model):
    num_variables = int(
        np.sum([tf.keras.backend.count_params(p) for p in model.trainable_weights]))
    return num_variables


def get_num_classes(model):
    num_classes = model.layers[-1].output_shape[1]
    return num_classes

def get_data_length(model):
    data_length = model.layers[0].input_shape[0][1]
    return data_length


def make_ddf(model, y, num_classes):
    ddf = np.array([[[(y[a] * delta(b, c) * delta(a, c)
                       - delta(b, c) * y[a] * y[c]
                       - delta(a, b) * y[a] * y[c]
                       - delta(a, c) * y[b] * y[c]
                       + 2 * y[a] * y[b] * y[c])
                      for c in range(num_classes)]
                     for a in range(num_classes)]
                    for b in range(num_classes)])

    return ddf


def make_f1_scalars(model, xbs, Hinv):
    num_classes = get_num_classes(model)
    data_length = get_data_length(model)

    start = time.time()
    f1_scalars = []
    for i, xb in enumerate(xbs):
        y = model(xb[:-1].reshape(1, -1)).numpy().reshape(-1)

        ddf = make_ddf(model, y, num_classes)

        Hinv_xb = np.tensordot(Hinv, xb, axes=([0], [0]))
        Hinv_xb = np.tensordot(Hinv_xb, xb, axes=([1], [0]))

        f1_scalar = np.tensordot(Hinv_xb, ddf, axes=([0, 1], [0, 1]))
        f1_scalars.append(f1_scalar)
        if i % 10000 == 0:
            print('time to make {} f1_scalars'.format(i), time.time() - start)
    return np.array(f1_scalars)


def make_G(model, xbs, Hinv):
    num_classes = get_num_classes(model)
    data_length = get_data_length(model)

    start = time.time()
    G = np.zeros([num_classes, data_length + 1])
    for i, xb in enumerate(xbs):
        x = xb[:-1].reshape(1, -1)
        y = model(x).numpy().reshape(-1)

        t1 = np.array([[[(y[a] * delta(b, c) * delta(a, c)
                          - delta(b, c) * y[a] * y[c]
                          - delta(a, b) * y[a] * y[c]
                          - delta(a, c) * y[b] * y[c]
                          + 2 * y[a] * y[b] * y[c])
                         for c in range(num_classes)]
                        for a in range(num_classes)]
                       for b in range(num_classes)])

        H1 = np.tensordot(Hinv, xb.reshape(1, data_length + 1), axes=([0], [1]))
        H1 = H1.reshape(num_classes, data_length + 1, num_classes)

        H2 = np.tensordot(H1, xb.reshape(1, data_length + 1), axes=([1], [1]))
        H2 = H2.reshape(num_classes, num_classes)

        t2 = np.tensordot(t1, H2, axes=([0, 1], [0, 1]))

        g = np.tensordot(t2, H1, axes=([0], [0]))
        g = np.swapaxes(g, 0, 1)
        G += g

        if (i + 1) % 5000 == 0:
            print('time to process {} samples for G:'.format(i + 1), time.time() - start)

    train_size = len(xbs)
    return train_size ** (-1) * G


def make_F(model, xb):
    num_classes = get_num_classes(model)

    y = model(xb[:-1].reshape(1, -1)).numpy().reshape(-1)
    t1 = np.array([[y[a] * (delta(a, b) - y[b])
                    for a in range(num_classes)]
                   for b in range(num_classes)])

    F = np.multiply.outer(t1, xb.reshape(-1))
    return F


def make_f2_scalars(model, xbs, X, Hinv):
    num_classes = get_num_classes(model)

    G = make_G(model, xbs, Hinv)

    start = time.time()
    f2_scalars = []
    for i, xb in enumerate(X):
        F = make_F(model, xb)
        f2_scalar = np.tensordot(F, G, axes=([1, 2], [0, 1]))
        f2_scalars.append(f2_scalar)
        if i % 5000 == 0:
            print('time to make {} f2_scalars'.format(i), time.time() - start)
    return np.array(f2_scalars)


def make_subleading_prediction(model, xbs, X, Hinv):
    data_length = get_data_length(model)

    f_scalars = model(np.array(X)[:, :-1].reshape(len(X), data_length)).numpy()
    f1_scalars = make_f1_scalars(model, X, Hinv)
    f2_scalars = make_f2_scalars(model, xbs, X, Hinv)

    return f_scalars, f1_scalars, f2_scalars


def make_truncated_hessian(model, h_evals, h_evecs, evalue_cutoff):
    num_classes = get_num_classes(model)
    data_length = get_data_length(model)

    evecs = np.transpose([evalue ** (-1 / 2) * h_evecs[:, i] for i, evalue in enumerate(h_evals)
                          if evalue > evalue_cutoff])
    num_evecs = evecs.shape[1]
    print('Number of eigenvectors:', num_evecs)
    Hinv = evecs @ np.transpose(evecs)
    Hinv = Hinv.reshape(data_length + 1, num_classes, data_length + 1, num_classes)

    return Hinv, num_evecs


def make_accs(y_true, y_pred):
    y_true = list(map(np.argmax, y_true))
    y_pred = list(map(np.argmax, y_pred))
    return acc(y_true, y_pred)


def make_total_predictions_from_outputs(outputs, num_train_samples, temp=1):
    f_scalars = outputs['f_scalars']
    f1_scalars = outputs['f1_scalars']
    f2_scalars = outputs['f2_scalars']
    output = f_scalars + temp * (2 * num_train_samples)**(-1) * (f1_scalars - f2_scalars)
    return output