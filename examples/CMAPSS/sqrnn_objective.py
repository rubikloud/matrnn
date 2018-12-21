import numpy as np
import keras.backend as K


class SQLoss(object):
    '''
    method 'loss' takes (ytrue, ypred)
    ytrue and ypred has same shape except last dimension
        ytrue has 3 while ypred has 2
    '''
        
    def loss(self, ytrue, ypred):
        
        # this is ytrue...
        # y[..., {0, 1, 2, 3}] is {tse, tte, uncensored, purchstatus}        
        tse = ytrue[..., 0]
        tte = ytrue[..., 1]
        unc = ytrue[..., 2]
        purchstatus = ytrue[..., 3]
        
        # marginalize by event type
        out = ypred[..., 0] - tte
        # sqloss
        out = K.pow(out, 2)
        # multiply by indicators for censoring and purchase status
        out = purchstatus * unc * out
        # marginalize by event type
        out = K.sum(out, axis=-1)
            
        return out