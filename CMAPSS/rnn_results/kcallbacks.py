import os
import keras
import numpy as np
from keras import backend as K

class SaveValidWeights(keras.callbacks.Callback):
    
    def __init__(self, weightsfname):
        self.weightsfname = weightsfname
        
    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        if not np.isnan(loss):
            self.model.save_weights(self.weightsfname)
            
            
class EarlyStopping(keras.callbacks.Callback):
    
    def __init__(self, patience=20, logsget='val_loss'):
        self.patience = patience
        self.loss = float('inf')
        self.logsget = logsget
        self.wait = 0
        
    def on_epoch_end(self, epoch, logs={}):
        losstemp = logs.get(self.logsget)
        # update tracker
        if losstemp < self.loss:
            self.loss = losstemp
            self.wait = 0
        elif losstemp == self.loss:
            self.wait += 1
        # stop training if wait beyond patience
        if self.wait > self.patience:
            print('early stopped!')
            self.model.stop_training = True
            
            
class TacticalRetreat(keras.callbacks.Callback):
    
    def __init__(self, weightsfname, lr_factor = .1, lr_min = np.finfo(float).eps):
        self.weightsfname = weightsfname
        self.lr_factor, self.lr_min = lr_factor, lr_min
        
    def loadlastgood(self):
        print ('\nretrieve last good weights...')
        if os.path.isfile(self.weightsfname):
            self.model.load_weights(self.weightsfname)
        else:
            print ('no last good weights, stop training')
            self.model.stop_training = True
        
    def on_epoch_end(self, epoch, logs={}):
        loss = logs.get('loss')
        if np.isnan(loss):
            print ('bad loss', loss)
            self.loadlastgood()
            
            # reduce learning rate
            lrnow = K.get_value(self.model.optimizer.lr)
            logs['lr'] = lrnow
            lrnew = max(self.lr_min, lrnow * self.lr_factor)
            K.set_value(self.model.optimizer.lr, lrnew)
            print ('lrnow: %e, lrnew: %e' % (lrnow, lrnew))
            
            if lrnew == self.lr_min:
                print ('lrnew at min, stop training!')
                self.model.stop_training = True
