import time
import os
import gzip
import pickle
import platform

import scipy
import numpy as np
import pandas as pd

from keras.optimizers import adam
from keras.models import Sequential
from keras.layers import Dense, Lambda
from keras.layers.core import Masking, Reshape
from keras.layers.wrappers import TimeDistributed
from keras.layers import Dense, GRU, LSTM
from keras.layers.core import Dropout

import matrnn_objective as obj
import matrnn_distributional as dist
from kcallbacks import SaveValidWeights, TacticalRetreat, EarlyStopping


class MATRNN(object):
    
    def __init__(self, modelspec_tuple, jobid, iswtte=False):

        self.modelspec_tuple = modelspec_tuple
        d, w = modelspec_tuple
        self.weightsfname = 'weights_jobid' + str(jobid) + '_d' + str(d) + 'w' + str(w) + '.h5'
        self.outputshape = (1, 2)
        self.iswtte = iswtte
        

    def compile(self, nvar, nseq, iniscale, lr=.01, verbose=1):

        d, w = self.modelspec_tuple
        nonlin = 'tanh'
        
        self.kmodel = Sequential()
        self.kmodel.add(Masking(mask_value=-1., input_shape=(None, nvar)))

        for k in range(d):
            self.kmodel.add(LSTM(w, return_sequences=True, dropout=.2))
        
        self.kmodel.add(Dense(np.prod(self.outputshape), activation=nonlin))
        self.kmodel.add(TimeDistributed(Reshape(self.outputshape)))
        
        # activation
        self.kmodel.add(Lambda(obj.activation, arguments={"iniscale": iniscale}))
        loss = obj.ExcessConditionalLoss(iswtte=self.iswtte).loss
        
        # compile model
        self.kmodel.compile(loss=loss, optimizer=adam(lr=lr, clipvalue=5.))
        self.kmodel.summary()

        
    def fit(self, xtrain, ytrain, iniscale, lr=.01, epochs=100, batch_size=1024, verbose=1):
        
        nobs, nseq, nvar = xtrain.shape
        self.compile(lr=lr, 
                     nvar=xtrain.shape[2], nseq=xtrain.shape[1], 
                     iniscale=iniscale)
        
        if verbose>0:
            print ('\nchecking if nans in data...')
            print ('nans in xtrain:\n', np.where(np.isnan(xtrain)))
            print ('nans in ytrain:\n', np.where(np.isnan(ytrain)))
        
            print ('\nchecking if activation is correct...')
            atrain = self.kmodel.predict(x=xtrain)
            print ('activation shape:', atrain.shape)

            print ('\nchecking if loss evaluation is valid...')
            print ('overall loss:', self.kmodel.evaluate(x=xtrain, y=ytrain, batch_size=batch_size, verbose=1))
        
        # callbacks
        save_val = SaveValidWeights(self.weightsfname)
        tact_ret = TacticalRetreat(self.weightsfname)
        earlystop = EarlyStopping(patience=20)
        
        t0 = time.time()        
        if verbose>0:
            print ('doing training...')
            
        self.kmodel.fit(xtrain, ytrain, validation_split = .1, shuffle = True,
                        batch_size = batch_size,
                        epochs = epochs, callbacks = [save_val, tact_ret, earlystop],
                        verbose = 2)
        
        if verbose>0:
            print ('training done in:', time.time()-t0)

        
    def infer(self, x, verbose=1):
        
        # run the model and keep final state...
        self.kmodel.load_weights(self.weightsfname)
        
        if verbose>0: 
            t0 = time.time()
            print ('running model...')
        out = self.kmodel.predict(x)
        
        if verbose>0:
            print ('inference done in:', time.time()-t0)
            
        return out