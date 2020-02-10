#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.lines as mlines
from scipy import interpolate

import lattice as lat
import constants as const
import operators as op


# In[3]:


ax = .1  #unit cell size along x-direction in [A]
ay = .1
Ny = 25    #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx


# In[4]:


R = 25
r = 10
donut = lat.donut(R, r) #donut coordinate array
NN_d = lat.NN_Arr(donut) #nearest neighbor array for donut
E0_d, states_d = op.diagH(donut, ax, ay) #energy eigenvalues and eigenvectors of donut lattice


# In[5]:


get_ipython().run_line_magic('matplotlib', 'notebook')
plt.scatter(donut[:,0], donut[:,1])
plt.show()


# In[6]:


get_ipython().run_line_magic('matplotlib', 'notebook')
#This is to visualize the array as points and see if the nearest neighbor array is working correctly

idx = 50
plt.scatter(donut[:, 0], donut[:, 1], c = 'b')
plt.scatter(donut[idx, 0], donut[idx, 1], c = 'r')
plt.scatter(donut[NN_d[idx,0],0], donut[NN_d[idx,0],1], c = 'g')
plt.scatter(donut[NN_d[idx,1],0], donut[NN_d[idx,1],1], c = 'magenta')
plt.scatter(donut[NN_d[idx,2],0], donut[NN_d[idx,2],1], c = 'purple')
plt.scatter(donut[NN_d[idx,3],0], donut[NN_d[idx,3],1], c = 'cyan')
plt.show()


# In[7]:


#Donut Eigenvalues
print(E0_d.shape)
print(E0_d[0:10]/E0_d[0])


# In[68]:


get_ipython().run_line_magic('matplotlib', 'notebook')
op.state_cplot(donut, states_d[:, 29])    


# In[ ]:




