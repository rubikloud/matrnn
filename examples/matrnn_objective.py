import numpy as np
import keras.backend as K


def single_activation(sc, sh, iniscale, maxshape = 10.):
    '''return tuple of activated values of (scale, shape) for single observation'''
    eps = K.epsilon()
    sc = iniscale*K.exp(sc)
    
    if maxshape>1.: 
        sh = sh-np.log(maxshape-1.)
        
    sh = maxshape*K.clip(K.sigmoid(sh), eps, 1-eps)
    
    return (sc, sh)


def activation(ypred, iniscale, maxshape = 10.):
    '''return tensor of activated outputs with last index being (scale, shape)
    prep nn output layer by converting it into natural scale
    scale:
        initialize scale assuming shape is 1, at mle
    shape:
        cap at maxshape
    note that output is of same shape as input x and this allows for hierarchical predictions
        e.g. x.shape = (nevents, 2)
    '''
    sc = ypred[..., 0]
    sh = ypred[..., 1]
    sc, sh = single_activation(sc, sh, iniscale, maxshape)
    return K.stack([sc, sh], axis=-1)


def single_loglike(tse, tte, sc, sh, iswtte=False):
    '''return discrete and right-censored loglikelihoods for single observation'''
    eps  = K.epsilon()
    haz0 = K.pow((tse+tte+eps)/sc, sh)
    haz1 = K.pow((tse+tte+1. )/sc, sh)
    hazc = K.pow((tse+eps    )/sc, sh)

    if iswtte:
        ## WTTE-RNN stuff:
        ## https://github.com/ragulpr/wtte-rnn/blob/master/python/wtte/wtte.py
        haz0 = K.pow((tte+eps)/sc, sh)
        haz1 = K.pow((tte+1. )/sc, sh)
        hazc = 0

    # for interval censored...
    # exp(-haz0)-exp(-haz1) = exp(-haz1)       (  exp(-haz0-(-haz1)) - 1 )
    # log(...)              = -haz1     +   log(  exp(-haz0-(-haz1)) - 1 )
    loglike_ivc             = -haz1     + K.log(K.exp(-haz0-(-haz1)) - 1.)
    # log(...)-log(S(tse))
    loglike_ivc = loglike_ivc - (-hazc) 

    # for right censored...
    # log(S(tse+tte))-log(S(tse))
    loglike_rc = -haz0 - (-hazc)
    
    return (loglike_ivc, loglike_rc)


class ExcessConditionalLoss(object):
    '''
    method 'loss' takes (ytrue, ypred)
    ytrue and ypred has same shape except last dimension
        ytrue has 3 while ypred has 2
    '''
    
    def __init__(self, iswtte=False):
        self.iswtte = iswtte
        
    def loss(self, ytrue, ypred):
        '''return loss as -1.*loglikelihood'''
        # this is ytrue...
        # y[..., {0, 1, 2, 3}] is {tse, tte, uncensored, purchstatus}        
        tse = ytrue[..., 0]
        tte = ytrue[..., 1]
        unc = ytrue[..., 2]
        purchstatus = ytrue[..., 3]
        
        sc = ypred[..., 0]
        sh = ypred[..., 1]
        
        llivc, llrc = single_loglike(tse, tte, sc, sh, iswtte=self.iswtte)
        loglike = unc*llivc + (1-unc)*llrc
        loglike = purchstatus*loglike
        
        # marginalize by event type
        loglike = K.sum(loglike, axis=-1)
            
        return -1.*loglike