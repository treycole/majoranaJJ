#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.lines as mlines
from scipy import interpolate

import lattice as lat
import constants as const
import operators as op


# In[2]:


ax = .1  #unit cell size along x-direction in [A]
ay = .1
Ny = 25    #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx


# In[3]:


R = 20


# In[4]:


CA = lat.halfdisk(R)
NN = lat.NN_Arr(CA)
energy, states = op.diagH(CA, ax, ay)


# In[5]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.scatter(CA[:,0], CA[:,1])
plt.xlim(0, max(CA[:,0]))
plt.ylim(0, max(CA[:,1]))
plt.show()


# In[8]:


get_ipython().run_line_magic('matplotlib', 'notebook')
#This is to visualize the array as points and see if the nearest neighbor array is working correctly
idx = 45
plt.scatter(CA[:,0],CA[:,1],c = 'b')
plt.scatter(CA[idx,0],CA[idx,1],c = 'r')
plt.scatter(CA[NN[idx,0],0], CA[NN[idx,0],1],c = 'g')
plt.scatter(CA[NN[idx,1],0], CA[NN[idx,1],1],c = 'magenta')
plt.scatter(CA[NN[idx,2],0], CA[NN[idx,2],1],c = 'purple')
plt.scatter(CA[NN[idx,3],0], CA[NN[idx,3],1],c = 'cyan')
plt.xlim(0, max(CA[:,0]))
plt.ylim(0, max(CA[:,1]))
plt.show()


# In[9]:


#Ibeam Eigenvalues
print(energy.shape)
print(energy[0:10]/energy[0])


# In[26]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.xlim(0, max(CA[:,0]))
plt.ylim(0, max(CA[:,1]))
op.state_cplot(CA, states[:, 2])


# In[ ]:




