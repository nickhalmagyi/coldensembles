import numpy as np
from tensorflow.keras.optimizers.schedules import LearningRateSchedule



class lr_scheduler(LearningRateSchedule):

    
    def __init__(self, initial_learning_rate):
        
        self.initial_learning_rate = initial_learning_rate
        
        self.epochs = [5,4,3,2,1]
        
        
    def __call__(self, step):
        num_epochs = len(self.epochs)
        
        for i, epoch in enumerate(self.epochs):
            if step > epoch:        
                return self.initial_learning_rate * 10**(num_epochs-i)
            
        return self.initial_learning_rate


    
class StepDecay():
    def __init__(self, initAlpha=0.01, factor=0.25, dropEvery=1):
        
        # store the base initial learning rate, drop factor, and
        # epochs to drop every
        self.initAlpha = initAlpha
        self.factor = factor
        self.dropEvery = dropEvery

        
    def __call__(self, epoch):
        
        # compute the learning rate for the current epoch
        exp = np.floor((1 + epoch) / self.dropEvery)
        alpha = self.initAlpha * (self.factor ** exp)

        # return the learning rate
        return float(alpha)