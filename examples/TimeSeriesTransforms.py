import numpy as np

def tse(indicators):
    '''return time since event, given vector of indicators'''
    ndat = len(indicators)
    tse = np.zeros(ndat)
    accum_tse = 0
    for tdex in range(ndat):
        if indicators[tdex] == 0:
            accum_tse += 1
        else:
            accum_tse = 0
        tse[tdex] = accum_tse
    return tse


def tte(indicators):
    '''return time to event, given vector of indicators'''
    ndat = len(indicators)
    tte = np.zeros(ndat)
    accum_tte = 0
    for tdex in range(ndat - 1, 0, -1):
        if indicators[tdex] == 0:
            accum_tte += 1
        else:
            accum_tte = 0
        tte[tdex - 1] = accum_tte
    return tte