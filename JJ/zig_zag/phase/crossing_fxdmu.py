import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sparse
import scipy.linalg as LA
import scipy.sparse.linalg as spLA
from scipy.signal import argrelextrema

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
from majoranaJJ.modules.gamfinder import gamfinder as gf
from majoranaJJ.modules.gamfinder import gamfinder_lowE as gfLE
from majoranaJJ.operators.potentials.barrier_leads import V_BL

#Defining System
Nx = 3 #Number of lattice sites along x-direction
Ny = 360 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 40 #Junction region
cutx = 0 #(Nx - 2*Sx) #width of nodule
cuty = 0 #0 #height of nodule

Junc_width = Wj*ay*.10 #nm
SC_width = ((Ny - Wj)*ay*.10)/2 #nm
Nod_widthx = cutx*ax*.1 #nm
Nod_widthy = cuty*ay*.1 #nm
print("Nodule Width in x-direction = ", Nod_widthx, "(nm)")
print("Nodule Width in y-direction = ", Nod_widthy, "(nm)")
print("Junction Width = ", Junc_width, "(nm)")
print("Supercondicting Lead Width = ", SC_width, "(nm)")

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

##############################################################

steps = 1000

alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
gz_i = 0.0 #parallel to junction: [meV], actually zero but avoiding degeneracy
gx = np.linspace(0, 2, steps)
phi = np.pi #SC phase differences, only want pi in this case
delta = 1.0 #Superconducting Gap: [meV]
V0 = 0 #Amplitude of potential : [meV]
mu = np.linspace(0, 7, steps) #Chemical Potential: [meV]
V = V_BL(coor, Wj = Wj, V0 = V0) #potential in normal region

##############################################################

MU = 6 #fixed mu value
k = 100
gi = 0
gf = 1.5
tol = 0.01

H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, cutx=cutx, cuty=cuty, V=V, mu=MU, gammaz=1e-5, alpha=alpha, delta=delta, phi=phi, qx=0.0001*(np.pi/Lx), periodicX=True) #gives low energy basis

eigs_0, vecs_0 = spLA.eigsh(H0, k=k, sigma=0, which='LM')
vecs_0_hc = np.conjugate(np.transpose(vecs_0)) #hermitian conjugate

H_G0 =  spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 0, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True) #Matrix that consists of everything in the Hamiltonian except for the Zeeman energy in the x-direction

H_G1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 1, alpha = alpha, delta = delta, phi = phi, qx = 0, periodicX = True) #Hamiltonian with ones on Zeeman energy along x-direction sites

HG = H_G1 - H_G0    #the proporitonality matrix for gamma-x, it is ones along the sites that have a gamma value

HG0_DB = np.dot(vecs_0_hc, H_G0.dot(vecs_0))
HG_DB = np.dot(vecs_0_hc, HG.dot(vecs_0))

gx = np.linspace(gi, gf, steps)
eig_arr = np.zeros((gx.shape[0]))

for i in range(gx.shape[0]):
    print(steps-i)
    #H = H_G0 + gx[i]*HG
    H_DB = HG0_DB + gx[i]*HG_DB
    #H_DB = np.dot(vecs_0_hc, H.dot(vecs_0))
    eigs_DB, U_DB = LA.eigh(H_DB)

    eig_arr[i] = eigs_DB[int(k/2)]

eig_min_idx = np.array(argrelextrema(eig_arr, np.less)[0])
#local minima indices
print(gx[eig_min_idx[:]])
sys.exit()
G_crit = []
for j in range(eig_min_idx.size):
    gx_c = gx[eig_min_idx[j]] #gamma value at local minima, first approx
    gx_c_lower = gx[eig_min_idx[j]-1] #gamma value one step behind minima
    gx_c_higher = gx[eig_min_idx[j]+1] #gamma value one step in front of minima
    gx_finer = np.linspace(gx_c_lower, gx_c_higher, steps) #refined gamma range

    eig_arr_finer = np.zeros(gx_finer.size) #new eigen value array that is higher resolution around local minima in first approximation
    for i in range(gx_finer.shape[0]):
        #H = H_G0 + gx[i]*H_G1 #Hamiltonian
        H_DB = HG0_DB + gx_finer[i]*HG_DB
        #H_DB = np.dot(vecs_0_hc, H.dot(vecs_0)) #change of basis, diff basis
        eigs_DB, U_DB = LA.eigh(H_DB)

        eig_arr_finer[i] = eigs_DB[int(k/2)] #k/2 -> lowest postive energy state

    eig_min_idx_finer = np.array(argrelextrema(eig_arr_finer, np.less)[0]) #new local minima indices
    eigs_local_min_finer = eig_arr_finer[eig_min_idx_finer] #isolating local minima
    #G_crit = np.ones((n_boundry, eigs_local_min_finder.size))
    for k in range(eigs_local_min_finer.size):
        if eigs_local_min_finer[k] < tol: #if effectively zero crossing
            G_crit.append(gx_finer[eig_min_idx_finer[k]]) #append critical gamma
            #print(gx_finer[eig_min_idx_finer[k]])
            print(gx_finer[eig_min_idx_finer[k]])
