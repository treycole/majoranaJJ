import numpy as np
import matplotlib.pyplot as plt

"""
Plotting the lattice neighbors and boundary neighbors
"""
def neyb(idx, coor, NN = None, NNb = None):
    if NN is not None:
        idx = 0 #idx
        plt.scatter(coor[:, 0], coor[:, 1] ,c = 'b')
        plt.scatter(coor[idx, 0], coor[idx, 1], c = 'r')
        if NN[idx, 0] != -1:
            plt.scatter(coor[NN[idx, 0], 0], coor[NN[idx, 0], 1], c = 'yellow')
        if NN[idx, 1] != -1:
            plt.scatter(coor[NN[idx,1], 0], coor[NN[idx, 1], 1], c = 'magenta')
        if NN[idx, 2] != -1:
            plt.scatter(coor[NN[idx,2], 0], coor[NN[idx, 2], 1], c = 'purple')
        if NN[idx, 3] != -1:
            plt.scatter(coor[NN[idx,3], 0], coor[NN[idx, 3], 1], c = 'cyan')
        plt.xlim(-1, max(coor[:,0])+1)
        plt.show()

    if NNb is not None:
        idx = 0 #idx
        plt.scatter(coor[:, 0], coor[:, 1] ,c = 'b')
        plt.scatter(coor[idx, 0], coor[idx, 1], c = 'r')
        if NNb[idx, 0] != -1:
            plt.scatter(coor[NNb[idx, 0], 0], coor[NNb[idx, 0], 1], c = 'yellow')
        if NNb[idx, 1] != -1:
            plt.scatter(coor[NNb[idx,1], 0], coor[NNb[idx, 1], 1], c = 'magenta')
        if NNb[idx, 2] != -1:
            plt.scatter(coor[NNb[idx,2], 0], coor[NNb[idx, 2], 1], c = 'purple')
        if NNb[idx, 3] != -1:
            plt.scatter(coor[NNb[idx,3], 0], coor[NNb[idx, 3], 1], c = 'cyan')
        plt.xlim(-1, max(coor[:,0])+1)
        plt.show()

"""
Plotting wavefunctions

def state_cplot(coor, states, title = 'Probability Density'):

    N = coor.shape[0]
    M=states.shape[0]/N
    prob_dens = []
    for i in np.arange(0, N):
        for i in range(0,M+1):
        prob_dens.append(np.square(abs(states[i])) + np.square(abs(states[i+N])))

    print(sum(prob_dens))
    plt.scatter(coor[:,0], coor[:,1], c = prob_dens)
    plt.xlim(0, max(coor[:, 0]))
    plt.ylim(0, max(coor[:, 1]))
    plt.title(title)
    plt.colorbar()
    plt.show()
"""

"""
Plots band diagrams
"""
def bands(eigarr, q, Lx, Ly, title = 'Band Structure'):
    for j in range(eigarr.shape[1]):
        #plt.plot(q, eigarr[:, j], c ='b', linestyle = 'solid')
        plt.scatter(q, eigarr[:, j], c ='b')
    x = np.linspace(-np.pi/Lx, np.pi/Lx+0.1*(np.pi/Lx))
    plt.plot(x, 0*x, c='k', linestyle='solid', lw=1)
    plt.xticks(np.arange(-np.pi/Lx, np.pi/Lx+0.1*(np.pi/Lx), (np.pi/Lx)), ('-π/Lx', '0', 'π/Lx'))
    plt.xlabel('k [1/A]')
    plt.ylabel('Energy [eV]')
    plt.title(title)
    plt.show()

"""
Plots a phase diagram of y vs x
"""
def phase(x, y, xlabel = ' ', ylabel = ' ', title = 'Phase Diagram'):
    for i in range(y.shape[1]):
        plt.plot(x, y[i], c = 'b', linestyle  = 'solid')
    line = np.linspace(0, max(x))
    plt.plot(line , 0*line, color = 'k', linestyle = 'solid', lw = 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def state_cplot(coor, vec, title='wavefunction'):
    vec_Sq=np.square(np.absolute(vec))
    num_div=int(vec.shape[0]/coor.shape[0])
    s=coor.shape[0]
    vec_proj=np.zeros(s)
    for n in range(num_div):
        vec_proj[:]=vec_proj[:] + vec_Sq[n*s:(n+1)*s]
    print(sum(vec_proj))
    plt.scatter(coor[:,0],coor[:,1],c=vec_proj,cmap='hot')
    plt.title(title)
    plt.show()
