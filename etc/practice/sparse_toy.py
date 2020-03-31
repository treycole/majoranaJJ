
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as Spar
import scipy.sparse.linalg as SparLinalg
np.set_printoptions(linewidth = 500)

def NN_Arr(coor):
    N = coor.shape[0]
    NN = -1*np.ones((N,4), dtype = 'int')
    xmax = max(coor[:,0])
    ymax = max(coor[:,1])
    Lx = xmax + 1
    Ly = ymax + 1

    for i in range(N):
        xi = coor[i, 0]
        yi = coor[i, 1]

        if (i-1) >= 0 and abs(xi - 1) >= 0 and abs(xi - coor[i-1, 0]) == 1 and abs(yi - coor[i-1, 1]) == 0:
            NN[i, 0] = i - 1
        if (i+1) < N and abs(xi + 1) <= xmax and abs(xi - coor[i+1, 0]) == 1 and abs(yi - coor[i+1, 1]) == 0:
            NN[i, 2] = i + 1
        for j in range(0, int(Lx)+1):
            if (i + j) < N and abs(yi + 1) <= ymax and abs(yi - coor[int(i + j), 1]) == 1 and abs(xi - coor[int(i + j), 0]) == 0:
                NN[i, 1] = i + j
            if (i - j) >= 0 and abs(yi - 1) >= 0 and abs(yi - coor[int(i - j), 1]) == 1 and abs(xi - coor[int(i - j), 0]) == 0:
                NN[i, 3]= i - j
    return NN

def kSq_gen(coor,ax,ay,NN):

    row = []; col = []; data = []
    N = coor.shape[0]
    for i in range(N):
        # A[i,i] = 2./ax**2 + 2./ay**2
        row.append(i); col.append(i); data.append(2./ax**2 + 2./ay**2)

        if NN[i,0] != -1: # checking if we have a left nearest neighbor
            row.append(NN[i,0]); col.append(i); data.append(-1./ax**2)

        if NN[i,2] != -1: # checking if we have a right nearest neighbor
            row.append(NN[i,2]); col.append(i); data.append(-1./ax**2)

        if NN[i,1] != -1: # Check top nearest neighbor
            row.append(NN[i,1]); col.append(i); data.append(-1./ay**2)

        if NN[i,3] != -1: # Check bottom nearest neighbor
            row.append(NN[i,3]); col.append(i); data.append(-1./ay**2)
    kSq_csc = Spar.csc_matrix((data,(row,col)),shape = (N,N),dtype = 'complex')
    return kSq_csc

a = 1.
ax = 1.; ay = 1.1
Nx = 100
Ny = 100

coor_arr = np.zeros((Nx*Ny,2))
counter = 0
for j in range(Ny):
    for i in range(Nx):
        coor_arr[counter,0] = i * a
        coor_arr[counter,1] = j * a
        counter += 1
NN = NN_Arr(coor_arr)
kSq = kSq_gen(coor_arr,ax,ay,NN)
plt.scatter(coor_arr[:,0],coor_arr[:,1])
plt.scatter(coor_arr[NN[35],0],coor_arr[NN[35],1],c = 'r')
plt.show()


H = 1. * kSq # should be multiplied by (hbar^2 / (2*m))
num = 5 # This is the number of eigenvalues and eigenvectors you want
sigma = 100 *0.001766 # This is the eigenvalue we search around
which = 'LM'
print "H shape: ", H.shape
eigs,vecs = SparLinalg.eigsh(H,k=num,sigma = sigma, which = which)
idx_sort = np.argsort(eigs)
eigs = eigs[idx_sort]
vecs = vecs[:,idx_sort]
print eigs[0]

### Plotting
for i in range(vecs.shape[1]):
    plt.scatter(coor_arr[:,0],coor_arr[:,1],c = np.square(np.absolute(vecs[:,i])),cmap = 'hot')
    plt.title('E/Eo = %.5f' % (eigs[i] / eigs[0] * 2.))
    plt.show()










































#asd
