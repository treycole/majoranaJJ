
import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps

import numpy as np
import majoranaJJ.plots as plots
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA

Nx = 15
Ny = 15
ax = 2
ay = 2
coor = shps.square(Nx, Ny)
NN = nb.NN_Arr(coor)
NNb = nb.NN_Bound(coor)

Wsc = Ny
Wj = 0
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gamma = 0.00  #Zeeman field energy contribution: [T]
delta = 0.01 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0 #Chemical Potential: [eV]

steps = 50 #Number of kx and ky values that are evaluated
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone



H=spop.HBDG(
       coor,ax,ay,NN,Wsc,Wj,mu=mu,delta=delta,gammaz=gamma,NNb=NNb, qx=qx[10])
num = 52 # This is the number of eigenvalues and eigenvectors you want
sigma = 0 *0.001766 # This is the eigenvalue we search around
which = 'LM'
print("H shape: ", H.shape)
"""
eigs,vecs = spLA.eigsh(H,k=num,sigma = sigma, which = which)
idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
vecs = vecs[:,idx_sort]
print(eigs[:])

plots.state_cplot(coor,vecs[:,10],title='10')
plots.state_cplot(coor,vecs[:,11],title='11')
plots.state_cplot(coor,vecs[:,12],title='12')
plots.state_cplot(coor,vecs[:,13],title='13')
"""
eig_arr=np.zeros((steps,num))
for i in range(steps):
    eig_arr[i,:]=np.sort(spLA.eigsh(spop.HBDG(
           coor,ax,ay,NN,Wsc,Wj,mu=mu,delta=delta,gammaz=gamma,
           NNb=NNb, qx=qx[i]),k=num,sigma=sigma,which=which)[0])

plots.bands(eig_arr,qx,Lx,Ly)
