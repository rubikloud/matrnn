import numpy as np

eps = np.finfo(float).eps


def gethaz(tse, tte, sc, sh):
    '''return tuple of hazards'''
    haz0 = np.power((tse+tte+eps)/sc, sh)
    haz1 = np.power((tse+tte+1. )/sc, sh)
    hazc = np.power((tse+eps    )/sc, sh)
    return haz0, haz1, hazc


def logsurv(elapsed, excess, scale, shape):
    '''return log of conditional survival function'''
    haz0, haz1, hazc = gethaz(elapsed, excess, scale, shape)
    return -haz0 - (-hazc)


def logdiscrete(elapsed, excess, scale, shape):
    '''return log of conditional discretized pmf'''
    haz0, haz1, hazc = gethaz(elapsed, excess, scale, shape)
    # for interval censored...
    # exp(-haz0)-exp(-haz1) = exp(-haz1)        (   exp(-haz0-(-haz1)) - 1  )
    # log(...)              = -haz1     +    log(   exp(-haz0-(-haz1)) - 1  )
    loglike_ivc             = -haz1     + np.log(np.exp(-haz0-(-haz1)) - 1.0)
    return loglike_ivc - (-hazc)


def logdense(elapsed, excess, scale, shape):
    '''return log of conditional density'''
    tse, tte, sc, sh = elapsed, excess, scale, shape    
    logsurvval = logsurv(tse, tte, sc, sh)
    return np.log(sh/sc) + (sh-1)*np.log((tse+tte+eps)/sc) + logsurvval


def quantile(elapsed, p, scale, shape):
    '''return quantile of conditional cdf'''
    _, _, hazc = gethaz(elapsed, 1, scale, shape)
    haz0 = hazc - np.log(1-p)
    out = scale * np.power(haz0 , 1/shape) - elapsed
    return out