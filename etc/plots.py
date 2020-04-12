import numpy as np
import matplotlib.pyplot as plt

"""
Plotting the lattice neighbors and boundary neighbors
"""
def lattice(idx, coor, NN = None, NNb = None):
    plt.scatter(coor[:, 0], coor[:, 1] , c = 'b')
    plt.scatter(coor[idx, 0], coor[idx, 1], c = 'r')

    if NN is not None:
        if NN[idx, 0] != -1:
            plt.scatter(coor[NN[idx, 0], 0], coor[NN[idx, 0], 1], c = 'yellow')
        if NN[idx, 1] != -1:
            plt.scatter(coor[NN[idx,1], 0], coor[NN[idx, 1], 1], c = 'magenta')
        if NN[idx, 2] != -1:
            plt.scatter(coor[NN[idx,2], 0], coor[NN[idx, 2], 1], c = 'purple')
        if NN[idx, 3] != -1:
            plt.scatter(coor[NN[idx,3], 0], coor[NN[idx, 3], 1], c = 'cyan')

    if NNb is not None:
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
    return

"""
Plots band diagrams
"""
def bands(eigarr, q, direction = 'x', units = '[eV]', title = 'Band Structure'):
    for i in range(eigarr.shape[1]):
        plt.plot(q, eigarr[:, i], c ='b', linestyle = 'solid')
        #plt.scatter(q, eigarr[:, i], c ='b')
    plt.plot(q, 0*q, c = 'k', linestyle='solid', lw=1)
    plt.xticks(np.linspace(min(q), max(q), 3), ('-π/Lx', '0', 'π/Lx'))
    plt.xlabel('k{} [1/A]'.format(direction))
    plt.ylabel('Energy {}'.format(units))
    plt.title(title)
    plt.show()

"""
Plots a phase diagram of y vs x
"""
def phi_phase(x, y, xlabel = 'Phi (SC Phase Difference)', ylabel = ' ', title = 'Phase Diagram'):
    for i in range(y.shape[1]):
        plt.plot(x, y[:, i], c = 'b', linestyle = 'solid')
    zeroLine = np.linspace(0, max(x))
    plt.plot(zeroLine , 0*zeroLine, color = 'k', linestyle = 'solid', lw = 1)
    plt.xticks(np.arange(0, 2*np.pi+1, np.pi/2), ('0', 'π/2', 'π', '3π/2', '2π'))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

def phase(x, y, xlabel = ' ', ylabel = ' ', title = 'Phase Diagram'):
    for i in range(y.shape[1]):
        plt.plot(x, y[:, i], c = 'b', linestyle = 'solid')
    zeroLine = np.linspace(0, max(x))
    plt.plot(zeroLine , 0*zeroLine, color = 'k', linestyle = 'solid', lw = 1)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()

"""
Plotting the probability density
"""
def state_cmap(coor, eigs, states, n = 1,  cmap = 'hot', title = 'Probability Density Map'):
    N = coor.shape[0]
    num_div = int(states.shape[0]/N)
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    states = states[:, idx_sort]

    probdens = np.square(abs(states[:, n]))
    map = np.zeros(N)
    for i in range(num_div):
        map[:] = map[:] + probdens[i*N : (i+1)*N]

    print(sum(map))
    plt.scatter(coor[:, 0], coor[:, 1], c = map, cmap = cmap)
    plt.title(title)
    plt.xlim(0, max(coor[:, 0]))
    plt.ylim(0, max(coor[:, 1]))
    plt.colorbar()
    plt.show()
