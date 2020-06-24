


import sys
import os
dir = os.getcwd()
os.chdir('..')
home_dir = os.getcwd()
sys.path.append('%s/Build_Files' % (home_dir) )   # Adding Build_Files to system path
os.chdir(dir)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
import Junction_Model_Class as JMC
import parameters as par
np.set_printoptions(linewidth = 500)

### parameters
Lx = 50. * 10.
W_sc = 2000. * 10.
W_j = 20. * 10.
W_c1 = 5. * 10.
W_c2 = 10. * 10.
L_c = 20. * 10.
a_SC = 2. * 10.
a_J = 1. * 10.
m_eff = 0.023
alpha = 250.

W_sc_buffer = 10. * 10.
ay_extended_targ = 20. * 10.




### Creating instance of two channel system
system = JMC.Junction_Model(Lx,W_sc,W_j,W_c1,W_c2,L_c,a_SC,a_J,m_eff,alpha,W_sc_buffer = W_sc_buffer, ay_extended_targ= ay_extended_targ)
system.MESH.PLOT.plot_elements2()
#system.MESH.PLOT.plot_mesh2()
#sys.exit()



### Calculating spectrum
V_j = 0.; V_sc = 85.; mu = 0.; Gam = 1.e-2

num = 400
qx_knot = .01 * np.pi/Lx
lNRG = system.HAM.generate_lNRG_subspace(qx_knot,V_j,V_sc,num)

qx = np.linspace(0.,np.pi/Lx * 1.,1001)
eig_arrL = np.zeros((qx.size,num))
for i in range(qx.size):
    if i % 10 == 0:
        print (qx.size - i)
    HamL = lNRG.compile_Ham(qx[i],mu,Gam,V_j,V_sc)
    eig_arrL[i,:],U = lNRG.solve_Ham(HamL)
for i in range(num):
    plt.plot(qx,eig_arrL[:,i],c = 'k')
    plt.plot(-qx,eig_arrL[:,i],c = 'k')
plt.grid()
plt.show()


"""
Ham,S = system.HAM.compile_Ham(0.,mu,Gam,V_j,V_sc)
print (Ham.shape)
eigs, U  = system.HAM.solve_Ham(Ham,S,200,0.,Return_vecs = True)
weight_junction = system.state_weight_junction(U)
E_crossing = eigs[0] + 2*1000.*par.hbm0*np.pi**2/(m_eff*Lx**2)
print (E_crossing)
#print (np.around(eigs[::2],decimals = 4))
#print (np.around(1. - weight_junction[::2],decimals = 4))
for j in range(int(eigs.size/2)):
    print (np.around(eigs[2*j],decimals = 4), 1. - np.around(weight_junction[2*j],decimals = 4))
    #if 1. - weight_junction[2*j] < .3:
    #    system.MESH.PLOT.PLOT_STATE3(U[:,2*j])
#sys.exit()
"""

"""
num = 50
qx = np.linspace(0.,np.pi/Lx * 1.,51)
eig_arr = np.zeros((qx.size,num))
eig_arrL = np.zeros((qx.size,num))
for i in range(qx.size):
    if i % 10 == 0:
        print (qx.size - i)
    Ham,S = system.HAM.compile_Ham(qx[i],mu,Gam,V_j,V_sc)
    if i == 0:
        print ("Ham shape: ", Ham.shape)
    eig_arr[i,:] = system.HAM.solve_Ham(Ham,S,num,0.)

    HamL = lNRG.compile_Ham(qx[i],mu,Gam,V_j,V_sc)
    eig_arrL[i,:],U = lNRG.solve_Ham(HamL,num = num)
for i in range(num):
    plt.plot(qx,eig_arr[:,i],c = 'k')
    plt.plot(-qx,eig_arr[:,i],c = 'k')
    plt.plot(qx,eig_arrL[:,i],c = 'r',linestyle = 'dashed')
    plt.plot(-qx,eig_arrL[:,i],c = 'r',linestyle = 'dashed')
plt.grid()
plt.show()
"""
