
import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.lines as mlines
from scipy import interpolate

π = np.pi
hbar = 6.58211899e-16           #h/2π in [eV*s]
m0 = 9.10938215e-31             #electron mass in [kg]
e0 = 1.602176487e-19            #electron charge in [C]
ηm = (hbar**2*e0*10**20)/m0     #hbar^2/mo in [eV A^2]
μB = 5.7883818066e-2            #in [meV/T]
meVpK = 8.6173325e-2            #Kelvin into [meV]
######################################################
ax = 100.0  #unit cell size along x-direction in [A]
ay = 100.0
Ny = 10     #number of lattice sites in y direction
Nx = 10     #number of lattice sites in x direction
N = Ny*Nx
#####################################################

#defining lattice, numbered 0->N
lattice = np.zeros((Nx, Ny))
for i in range(Ny):
    for j in range(Nx):
        lattice[i, j] = j + i*Ny

#defining coordinate array
#coordinate array is Nx2, first column is array of x values in units of [A], second column is y values in units of [A]
coor = np.zeros((N,2))
    for i in range(Nx):
        for j in range(Ny):
            n = i + Nx * j
            x = (i) * ax
            y = (j) * ay
            coor[n,0] = x
            coor[n,1] = y

#defining nearest neighbor array
#NN_arr is Nx4, the columns store the index of the nearest neighbors.
#Left: NN[n,0] = (n-Nx), Above: NN[n,1] = n+1, Right: NN[n, 2] = n+1, Down NN[n, 3] = n+Nx
#if there is no lattice site in nearest neighbor spot, value is -1
def NN_Arr(coor, ax, ay):
    N = coor.shape[0]
    tol = 1e-8
    NN = -1*np.ones((N,4), dtype = 'int')
    for n in range(N):
        for m in range(N):
            xn = coor[n, 0]
            xm = coor[m, 0]
            yn = coor[n,1]
            ym = coor[m,1]

            if abs((xn - xm) - ax)< tol and abs(yn - ym) < tol:
                NN[n, 0] = m
            if abs((xn - xm) + ax) < tol and abs(yn - ym) < tol:
                NN[n, 2] = m
            if abs((yn - ym) + ay) < tol and abs(xn - xm) < tol:
                NN[n, 1] = m
            if abs((yn - ym) - ay) < tol and abs(xn - xm) < tol:
                NN[n, 3]= m
    return NN

# Descritizing $k_x$ and $k_y$
def k_x(coor, ax, ay):
    k_x = np.zeros((N,N), dtype = "complex")
    NN = NN_Arr(coor, ax, ay)
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                k_x[j,i] = -1j/(2*ax)
            if NN[j, 2] == i:
                k_x[j,i] = 1j/(2*ax)
    return k_x
def k_y(coor, ax, ay):
    k_y = np.zeros((N,N), dtype = "complex")
    NN = NN_Arr(coor, ax, ay)
    for i in range(N):
        for j in range(N):
            if NN[j,1] == i:
                k_x[j,i] = 1j/(2*ax)
            if NN[j, 3] == i:
                k_x[j,i] = -1j/(2*ax)
    return k_x

#Descritizing kx^2 and ky^2
def k_x2(coor, ax, ay):
    k_x2 = np.zeros((N,N), dtype='complex')
    NN = NN_Arr(coor, ax, ay)
    for i in range(N):
        for j in range(N):
            if NN[j,0] == i:
                k_x2[j,i] = -1/ax**2
            if NN[j, 2] == i:
                k_x2[j,i] = -1/ax**2
            if i == j:
                k_x2[j,i] = 2/ax**2
    return k_x2

#Defining Hamiltonian for simple free particle in the lattice
def H0(coor, ax, ay):
    H = np.zeros((N,N), dtype = 'complex')
    H = hbar**2/(2*m0)*k_x2(coor, ax, ay)
    return H

#Getting energies
def E0(coor, ax, ay):
    H = H0(coor, ax, ay)
    eigvals, eigvecs = LA.eigh(H)
    return np.sort(eigvals)

#Getting States
def eigstate(coor, ax, ay):
     H = H0(coor, ax, ay)
     eigvals, eigvecs = LA.eigh(H)
     return eigvecs

NN =   NN_Arr(coor, ax, ay)
idx = 24
plt.scatter(coor[:,0],coor[:,1],c = 'b')
plt.scatter(coor[idx,0],coor[idx,1],c = 'r')
plt.scatter(coor[NN[idx,0],0],coor[NN[idx,0],1],c = 'g')
plt.scatter(coor[NN[idx,1],0],coor[NN[idx,1],1],c = 'magenta')
plt.scatter(coor[NN[idx,2],0],coor[NN[idx,2],1],c = 'purple')
plt.scatter(coor[NN[idx,3],0],coor[NN[idx,3],1],c = 'cyan')
plt.show()
