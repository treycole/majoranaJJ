import numpy as np
import matplotlib.pyplot as plt

import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.modules.plots as plots #plotting functions
#from majoranaJJ.modules.mufinder import mufinder

Nx = 4 #Number of lattice sites along x-direction
Ny = 80 #Number of lattice sites along y-direction
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Wj = 16  #Junction region
print("Junction Width = ", Wj*ay*.10, "(nm)")
print("Supercondicting Lead Width = ", ((Ny - Wj)*ay*.10)/2, "(nm)")

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)
lat_size = coor.shape[0]
print("Lattice Size: ", lat_size)

Lx = (max(coor[:, 0]) - min(coor[:, 0]) + 1)*ax #Unit cell size in x-direction
Ly = (max(coor[:, 1]) - min(coor[:, 1]) + 1)*ay #Unit cell size in y-direction

steps = 20

alpha = 100 #Spin-Orbit Coupling constant: [meV*A]
gammaz = 0 #Zeeman field energy contribution: [meV]
gammax = np.linspace(0, 0.4, steps) #parallel to junction: [meV]
phi = [0, np.pi] #SC phase difference
delta = 0.15 #Superconducting Gap: [meV]
V0 = 0.0 #Amplitude of potential : [meV]
mu = np.linspace(75, 81, steps) #Chemical Potential: [meV]

mu_pi = []
gamx_pi = []
mu_0 = []
gamx_0 = []

tlrnce = 1e-5
for i in range(steps):
    print(steps - i)
    for j in range(steps):
        energy_p0 = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, mu=mu[j], gammax=gammax[i], alpha=alpha, delta=delta, phi=phi[0], periodicX=True, k=2, maxiter=300)

        energy_ppi = spop.EBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, mu=mu[j], gammax=gammax[i], alpha=alpha, delta=delta, phi=phi[1], periodicX=True, k=2, maxiter=300)

        energy_p0, energy_ppi = np.array(energy_p0), np.array(energy_ppi)

        if abs(energy_p0[0] - energy_p0[1]) <= tlrnce:
            plt.scatter(gammax[i], mu[j], c='b')
            mu_0.append(mu[j])
            gamx_0.append(gammax[i])

        #if abs(energy_ppi[0] - energy_ppi[1]) <= tlrnce:
            #mu_bnd2.append(mu[j])
            #gamx_bnd2.append(gammax[i])

#plt.scatter(gamx_bnd2, mu_bnd2, label = r'$\phi = \pi$')
#plt.scatter(gamx_bnd1, mu_bnd1, label = r'$\phi = 0$')

plt.xlabel(r'$E_z$ (meV)')
plt.ylabel(r'$\mu$ (meV)')

plt.xlim(min(gammax), max(gammax))

plt.savefig('mu_gx.png')
plt.legend()
plt.show()
