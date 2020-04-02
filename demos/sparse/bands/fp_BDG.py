import time
import majoranaJJ.operators.sparsOP as spop
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.lattice.shapes as shps

import numpy as np
import majoranaJJ.plots as plots
import matplotlib.pyplot as plt
import scipy.sparse.linalg as spLA

Nx = 15
Ny = 15
ax = 2 #[A]
ay = 2 #[A]
coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_Arr(coor)
NNb = nb.NN_Bound(coor)

Wsc = Ny #Superconducting region
Wj = 0  #Junction region
Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

alpha = 0.0   #Spin-Orbit Coupling constant: [eV*A]
gamma = 0.00  #Zeeman field energy contribution: [T]
delta = 0.00 #Superconducting Gap: [eV]
V0 = 0.0 #Amplitude of potential : [eV]
mu = 0.0#Chemical Potential: [eV]

steps = 50 #Number of kx and ky values that are evaluated
qx = np.linspace(-np.pi/Lx, np.pi/Lx, steps) #kx in the first Brillouin zone
qy = np.linspace(-np.pi/Ly, np.pi/Ly, steps) #ky in the first Brillouin zone

#Hamiltonian for size test
H = spop.HBDG(coor,ax,ay, NN, Wsc, Wj)
num = 20 # This is the number of eigenvalues and eigenvectors you want
sigma = 0 *0.001766 # This is the eigenvalue we search around
which = 'LM'
print("H shape: ", H.shape)

"""
eig_arr = np.zeros((steps,num))
for i in range(steps):
    eig_arr[i,:]=np.sort(spLA.eigsh(spop.HBDG(
           coor,ax,ay,NN,Wsc,Wj,mu=mu,delta=delta,gammaz=gamma,
           NNb=NNb, qx=qx[i]),k=num,sigma=sigma,which=which)[0])

plots.bands(eig_arr, qx, Lx, Ly)
