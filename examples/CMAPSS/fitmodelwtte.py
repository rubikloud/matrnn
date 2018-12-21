
# coding: utf-8

# In[1]:


import os
import gzip
import pickle
import scipy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import utils
import matrnn_fitter as fitter


# In[2]:


epochs = 10000
lr = 0.00001
batch_size = 1024*16
d = 2
w = 64
winlen = 78
jobid = 'wtte'


# In[3]:


mtrain = pickle.load(gzip.open('mlocaltrain.pkl', 'rb'))
xtrain, ytrain = utils.split(mtrain)
xtrainlong, ytrainlong = utils.getlongver(xtrain, ytrain, winlen)
print ('xtrainlong.shape:', xtrainlong.shape)
print ('ytrainlong.shape:', ytrainlong.shape)


# In[4]:


# compute iniscale taking mean time to occurence (i.e. the tse at time-1) for non-windowed observations
iniscale = ytrain[:, -1, 0, 0]
iniscale = np.mean(iniscale)
print ('iniscale:', iniscale)


# In[5]:


model = fitter.MATRNN(modelspec_tuple=(d, w), jobid=jobid, iswtte=True)
model.fit(xtrainlong, ytrainlong, iniscale=iniscale, epochs=epochs, batch_size=batch_size, lr=lr)


# In[ ]:


mtest = pickle.load(gzip.open('mlocaltest.pkl', 'rb'))
print ('mtest.shape:', mtest.shape)
print ('mtest[0, :, :8]:\n', mtest[0, :, :8])
print ('mtest[0, :, -8:]:\n', mtest[0, :, -8:])
print ('np.all(mtest==-1., axis=2)[0,...]:\n', np.all(mtest==-1., axis=2)[0,...])


# In[ ]:


mtest = mtest[:,-winlen:,...]
xtest, ytest = utils.split(mtest)
inferred = model.infer(x=xtest)


# In[ ]:


finalstate = inferred[:, -1, 0, :]
finalstate_fname = 'finalstate_jobid' + str(jobid) + '_d' + str(d) + 'w' + str(w) + '.pkl'
pickle.dump(finalstate, gzip.open(finalstate_fname, 'wb'))