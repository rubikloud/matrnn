import numpy as np
import keras.backend as K


def single_activation(sc, sh, iniscale, maxshape = 10.):
    '''return tuple of activated values of (scale, shape) for single observation

    arguments:
        sc: tensor of scale output
        sh: tensor of shape output
        iniscale: float of initial value for scale
        maxshape: maximum shape parameter

    return:
        tuple of tensors for activated scale and shape
    '''

    eps = K.epsilon()
    sc = iniscale*K.exp(sc)
    
    if maxshape>1.: 
        sh = sh-np.log(maxshape-1.)
        
    sh = maxshape*K.clip(K.sigmoid(sh), eps, 1-eps)
    
    return (sc, sh)


def activation(ypred, iniscale, maxshape = 10.):
    '''return tensor of activated outputs with last index being (scale, shape)

    arguments:
        ypred: tensor of outputs
    	iniscale: initial value for scale (mle for case where shape is 1)
    	maxshape: maximum shape parameter

    return:
        tensor of stacked activated scale and shape
    '''

    sc = ypred[..., 0]
    sh = ypred[..., 1]
    sc, sh = single_activation(sc, sh, iniscale, maxshape)
    return K.stack([sc, sh], axis=-1)


def single_loglike(tse, tte, sc, sh, iswtte=False):
    '''return discrete and right-censored loglikelihoods for single observation

    arguments:
        tse: tensor of time since last event
        tte: tensor of time to next event (or end of training)
        sc: tensor of scale parameters
        sh: tensor of shape parameters
        iswtte: bool of whether model is WTTE

    return:
        tuple of tensors for uncensored and censored cases of log-likelihood
    '''

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
    
    def __init__(self, iswtte=False):
        self.iswtte = iswtte
        
    def loss(self, ytrue, ypred):
        '''return loss as negative of log-likelihood

        arguments:
            ytrue: tensor of (tse, tte, unc, purchstatus)
            ypred: tensor of activated neural network outputs

        return:
            tensor of loss
        '''

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