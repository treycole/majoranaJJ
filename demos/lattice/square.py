import matplotlib.pyplot as plt

import majoranaJJ.modules.constants as const
import majoranaJJ.modules.shapes as shp
import majoranaJJ.modules.neighbors as nb

Ny = 25    #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx
print(N)

coor = shp.square(Nx, Ny) #square coordinate array
NN =  nb.NN_Arr(coor) #nearest neighbor array of square lattice
NNk = nb.NN_Bound(NN, coor)

nbr = 1
plt.scatter(coor[:, 0],coor[:, 1], c = 'b')
plt.scatter(coor[nbr, 0],coor[nbr, 1], c = 'r')

if NN[nbr, 0] != -1:
    plt.scatter(coor[NN[nbr, 0], 0], coor[NN[nbr, 0], 1], c = 'green')
if NN[nbr, 1] != -1:
    plt.scatter(coor[NN[nbr,1], 0], coor[NN[nbr, 1], 1], c = 'magenta')
if NN[nbr, 2] != -1:
    plt.scatter(coor[NN[nbr,2], 0], coor[NN[nbr, 2], 1], c = 'purple')
if NN[nbr, 3] != -1:
    plt.scatter(coor[NN[nbr,3], 0], coor[NN[nbr, 3], 1], c = 'cyan')
plt.show()

nbr = 15
print(NN[nbr, 0], NN[nbr, 1], NN[nbr, 2], NN[nbr, 3])
plt.scatter(coor[:, 0], coor[:, 1] ,c = 'b')
plt.scatter(coor[nbr, 0], coor[nbr, 1], c = 'r')
if NNk[nbr, 0] != -1:
    plt.scatter(coor[NNk[nbr, 0], 0], coor[NNk[nbr, 0], 1], c = 'green')
if NNk[nbr, 1] != -1:
    plt.scatter(coor[NNk[nbr,1], 0], coor[NNk[nbr, 1], 1], c = 'magenta')
if NNk[nbr, 2] != -1:
    plt.scatter(coor[NNk[nbr,2], 0], coor[NNk[nbr, 2], 1], c = 'purple')
if NNk[nbr, 3] != -1:
    plt.scatter(coor[NNk[nbr,3], 0], coor[NNk[nbr, 3], 1], c = 'cyan')
plt.show()
