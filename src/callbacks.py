from keras.callbacks import Callback
import numpy as np



class EvaluateAfterNBatch(Callback):
    """
    A custom callback class which will evaulate data after each batch has run.
    """
    def __init__(self, X, Y, N=1):
        self.batch_count = 1
        self.N = N
        self.X, self.Y = X, Y
        self.true_preds = []
        self.false_preds = []

    def get_truefalse_preds(self):
        Y_true = np.array(list(map(np.argmax, self.Y)))

        y_preds = self.model.predict(self.X)

        Y_preds = np.array(list(map(np.argmax, y_preds)))
        Y_preds_prob = np.array(list(map(np.max, y_preds)))

        true_args = np.where((Y_preds == Y_true) == True)[0]
        false_args = np.where((Y_preds == Y_true) == False)[0]

        true_preds = np.array(Y_preds_prob[true_args])
        false_preds = np.array(Y_preds_prob[false_args])

        return true_preds, false_preds


    def on_batch_end(self, batch, logs={}):
        self.batch_count += 1
        if self.batch_count % self.N == 0:
            true_preds, false_preds = self.get_truefalse_preds()
            self.true_preds += [true_preds]
            self.false_preds += [false_preds]

            
class ModelMinTrainLoss(Callback):
    
    def __init__(self, X, Y, N=1):
        self.batch_count = 1
        self.N = N
        self.X, self.Y = X, Y
        self.true_preds = []
        self.false_preds = []
    
