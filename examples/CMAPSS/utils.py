import numpy as np


def split(m):
    '''
    xtrain, ytrain = split(m)
    '''
    nobs, nseq, nvar = m.shape
    # (tse, tte, unc, pcs)
    # indices at [1,2] are tte, unc 
    x = m[:, :, np.delete(np.arange(nvar), [1,2])]
    
    y = m[:, :, :4]
    # cap tte at 130 for cmapss datasets
    # http://www.hitachi-america.us/rd/about_us/bdl/docs/LSTM_RUL.PDF
    #y[:, :, 1] = np.minimum(y[:, :, 1], 130 * np.ones((nobs, nseq)))
    y = y.reshape((nobs, nseq, 1, 4))
    
    y[y<0] = 0
    return x, y


def windowed(xsingle, ysingle, winlen):
    '''
    split into windowed
    '''
    # xsingle.shape: nseq, ncov
    # ysingle.shape: nseq, 1, 4
    nseq, ncov = xsingle.shape
    resl = [(xsingle[start:(start+winlen), ...].reshape((1, winlen, ncov)), 
             ysingle[start:(start+winlen), ...].reshape((1, winlen, 1, 4)))
            for start in range(0, nseq-winlen)]
    return resl


def getlongver(xtrain, ytrain, winlen):
    '''
    window+concatenate
    '''

    resl = list(map(lambda i: windowed(xtrain[i, ...], ytrain[i, ...], winlen), np.arange(xtrain.shape[0])))

    reslall = []
    for resltemp in resl:
        reslall += resltemp

    xtrainlong = np.concatenate([reslall[i][0] for i in range(len(reslall))])
    ytrainlong = np.concatenate([reslall[i][1] for i in range(len(reslall))])
    
    notmasked = xtrainlong[:, 0, 0] > -1.
    xtrainlong = xtrainlong[notmasked, ...]
    ytrainlong = ytrainlong[notmasked, ...]
    
    return xtrainlong, ytrainlong