# Majorana Modes in two dimensional proximatized semiconductor/superconductor heterostrucutre
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as scl
# Values of parameters
Nx=4# Nx = Number of lattice points along x-axis
Ny=4# Ny=Number of lattice points along y-axis
N=Nx*Ny # N=Nx*Ny, total number of lattice sites
ax=500 # lattice constant along x-axis,define later(in 10e-10meter)
ay=500 # lattice constant, alog y-axix,define latter(in 10e-10 meter)
Lx=50e-9 # Length of lattice along x-axis (in meter)
Ly=50e-9 # Length lattiice along y-axis( in meter)
m=9.1e-31 #mass of electron in Kg
hbar= 6.05e-19 # h/2*pi  in eV-s
import numpy as np
import matplotlib.pyplot as plt
R=15
r=3
ax=1
ay=1
Nx=int(2*R/ax+1)
Ny=int(2*R/ay+1)
coor_arr_x=[]
coor_arr_y=[]
xmin=-R
ymin=-R
for j in range(Ny):
    for i in range (Nx):
        x=xmin+i*ax
        y=ymin+j*ay
        rij=np.sqrt(x**2+y**2)
        if rij<R and rij>r:
            coor_arr_x.append(x)
            coor_arr_y.append(y)

coor_arr=np.zeros((len(coor_arr_x),2))
coor_arr[:,0]=np.array(coor_arr_x)
coor_arr[:,1]=np.array(coor_arr_y)
plt.scatter(coor_arr[:,0],coor_arr[:,1],c='blue', s=50)
plt.show()
def neig_arr(coor_arr,ax,ay):
        tol = 10e-8
        N=coor_arr.shape[0]
        neig_arr=-1*np.ones((N,4),dtype="int")
        for n in range(N):
            for m in range(N):
                xn=coor_arr[n,0]
                yn=coor_arr[n,1]
                xm=coor_arr[m,0]
                ym=coor_arr[m,1]
                if abs((xn-xm)-ax)<tol and abs(yn-ym)<tol:
                    neig_arr[n,0]=m
                if abs((xn-xm)+ax)<tol and abs(yn-ym)<tol:
                    neig_arr[n,2]=m
                if abs((yn-ym)-ay)<tol and abs(xn-xm)<tol:
                    neig_arr[n,3]=m
                if abs((yn-ym)+ay)<tol and abs(xn-xm)<tol:
                    neig_arr[n,1]=m
        return neig_arr
# Lattice plot with earest neighbours
neig = neig_arr(coor_arr, ax, ay)
idx = 55 # index variable
_xpoints=coor_arr[:,0] # lattice points along x-axis
_ypoints=coor_arr[:,1] # lattice points along
plt.scatter(_xpoints,_ypoints,color = 'blue', s=50)
plt.scatter(coor_arr[idx,0],coor_arr[idx,1],c = 'red',s=50)
plt.scatter(coor_arr[neig[idx,0],0],coor_arr[neig[idx,0],1],c = 'green',s=50)
plt.scatter(coor_arr[neig[idx,1],0],coor_arr[neig[idx,1],1],c = 'magenta',s=50)
plt.scatter(coor_arr[neig[idx,2],0],coor_arr[neig[idx,2],1],c = 'purple',s=50)
plt.scatter(coor_arr[neig[idx,3],0],coor_arr[neig[idx,3],1],c = 'cyan',s=50)
print (coor_arr[-1])
print (neig[8,0])
plt.show()
# Discritization of Operators (kx,k**2x, ky, k**2y)
# Discritization of Kx
def kx(coor_arr,ax,ay):
    N=coor_arr.shape[0]
    kx=np.zeros((N,N),dtype="complex")
    for i in range(N):
        for j in range(N):
            if neig[j,0]==i:
                kx[j,i] = -1j/2*ax    # j in the right side is the imaginary uite
            if neig[j,2]==i:
                kx[j,i] = j/2*ax
        return kx

# Discritization of Ky
def ky(coor_arr,ax,ay):
    N=coor_arr.shape[0]
    ky=np.zeros((N,N),dtype="complex")
    for i in range(N):
        for j in range(N):
            if neig[j,0]==i:
                kx[j,i] = -1j/2*ay    # j in the right side is the imaginary uite
            if neig[j,2]==i:
                kx[j,i] = 1j/2*ay
        return ky

# Discritization of Kx^2
def kx_2(coor_arr, ax,ay):
    N=coor_arr.shape[0]
    kx_2=np.zeros((N,N),dtype="complex")
    for i in range(N):
        for j in range(N):
            if neig[j,0]==i:
                kx_2[j,i] = -1/ax**2
            if neig[j,2]==i:
                kx_2[j,i] = -1/ax**2
            if i==j:
                kx_2[j,i] = 2/ax**2
    return kx_2
B=kx_2(coor_arr, ax,ay)
print("B:",B)
#Discritization of Ky^2
def ky_2(coor_arr,ax,ay):
    N=coor_arr.shape[0]
    ky_2=np.zeros((N,N),dtype="complex")
    for i in range(N):
        for j in range(N):
            if neig[j,1]==i:
                ky_2[j,i] =- 1/ay**2
            if neig[j,3]==i:
                ky_2[j,i] = -1/ay**2
            if i==j:
                ky_2[j,i] = 2/ay**2
    return ky_2
# Define Hamilton
def Ham_gen(coor_arr, ax, ay):
    H=np.ones((N,N),dtype='complex')
    H=1.*(hbar)**2/(2**m)*(kx_2(coor_arr,ax,ay) + ky_2(coor_arr, ax,ay))
    return(H)
Ham = Ham_gen(coor_arr,ax,ay)
print (Ham)
# Find eigenvalues and eigenvectors
E,phi=np.linalg.eigh(Ham)
#print(E)
n = 6
phi_0= np.square(np.absolute(phi[:,n]))
plt.scatter(coor_arr[:,0], coor_arr[:,1], c =phi_0 , s=50)
plt.colorbar()
plt.show()
