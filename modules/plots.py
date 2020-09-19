import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt

"""
Plotting the lattice neighbors and boundary neighbors
"""
def lattice(
    idx, coor, NN = None, NNb = None,
    savenm = None
    ):

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
    if savenm is not None:
        plt.savefig(savenm)
    plt.show()

def junction(coor, delta, title = None, savenm = None):

    delta = delta.toarray()
    N = coor.shape[0]
    D = delta[0:N, N:]
    fig = plt.figure()
    axx = fig.add_subplot(1,1,1)

    axx.scatter(coor[:, 0], coor[:, 1] , c = 'k')
    for i in range(N):
        if np.any(np.abs(D[i, :])) != 0:
            if np.imag(D[i,i]) > 0:
                axx.scatter(coor[i, 0], coor[i, 1], c = 'b')
            if np.imag(D[i,i]) < 0:
                axx.scatter(coor[i, 0], coor[i, 1], c = 'g')
            if np.imag(D[i,i]) == 0:
                axx.scatter(coor[i, 0], coor[i, 1], c = 'r')
    axx.set_xlim(-5, max(coor[:,0])+5)
    axx.set_title('Green is negative phase argument, Blue is positive phase argument, Red is zero phase', wrap = True)
    axx.set_aspect(1.0)
    if savenm is not None:
        plt.savefig(savenm)

    plt.show()

def potential_profile(coor, V):
    V = sparse.bmat([[None, V], [V, None]], format='csc')
    junction(coor, V, title = 'Potential Profile')

"""
Plotting the probability density
"""
def state_cmap(
    coor, eigs, states,
    n = 1, cmap = 'hot',
    title = r'$|\psi|^2$',
    savenm = None
    ):

    N = coor.shape[0]
    num_div = int(states.shape[0]/N)
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    states = states[:, idx_sort]

    probdens = np.square(abs(states[:, n]))
    map = np.zeros(N)
    for i in range(num_div):
        map[:] = map[:] + probdens[i*N : (i+1)*N]

    print("Sum of prob density", sum(map))
    #plt.scatter(coor[:, 0], coor[:, 1], c = map, cmap = cmap)
    #plt.title(title)
    #plt.xlim(0, max(coor[:, 0]))
    #plt.ylim(0, max(coor[:, 1]))
    #plt.colorbar()
    fig = plt.figure()
    axx = fig.add_subplot(1,1,1)

    tc = axx.tricontourf(coor[:,0], coor[:,1], map, 1000)
    #axx.set_aspect(1.0)
    axx.set_title(title)
    if savenm is not None:
        plt.savefig(savenm)
    print("Energy Value of State", eigs[n])
    plt.show()

"""
Plots band diagrams
"""
def bands(
    k, eigarr,
    direction = 'x', units = 'meV',
    title = 'Bands',
    xlim = None, ylim = None,
    savenm = None
    ):

    for i in range(eigarr.shape[1]):
        plt.plot(k, eigarr[:, i], c ='mediumblue', linestyle = 'solid')
        plt.plot(-k, eigarr[:, i], c ='mediumblue', linestyle = 'solid')
        #plt.scatter(q, eigarr[:, i], c ='b')
    plt.plot(k, 0*k, c = 'k', linestyle='solid', lw=1)
    plt.plot(-k, 0*k, c = 'k', linestyle='solid', lw=1)
    #plt.xticks(np.linspace(min(k), max(k), 3), ('-π/Lx', '0', 'π/Lx'))
    plt.xlabel('k{} (1/A)'.format(direction))
    plt.ylabel('Energy ({})'.format(units))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.title(title, wrap = True)
    if savenm is not None:
        plt.savefig(savenm)
    plt.show()


"""
Plots a phase diagram of y vs x
"""
def phase(
    x, y,
    xlabel = ' ', ylabel = ' ',
    title = 'Phase Diagram',
    xlabels = None, xticks = None,
    xlim = None, ylim = None,
    label = None,
    savenm = None
    ):

    for i in range(y.shape[1]):
        plt.plot(x, y[:, i], c = 'mediumblue', linestyle = 'solid', label = label)
    #zeroLine = np.linspace(0, max(x))
    plt.xticks(xticks, xlabels)
    plt.plot(x , 0*x, color = 'grey', linestyle = 'solid', lw = 1)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if savenm is not None:
        plt.savefig(savenm)
    plt.legend()
    plt.show()

def phi_phase(
    phi, energy,
    Ez = 0,
    xlim = None, ylim = None,
    savenm = None
    ):

    N = energy.shape[1]
    red_band = energy[:, int(N/2)-1: int(N/2)+1]
    blue_band = np.array(energy)
    blue_band[:, int(N/2)-1: int(N/2)+1] = None

    for i in range(red_band.shape[1]):
        plt.plot(phi, red_band[:,i], c = 'red', ls = 'solid')
    for i in range(blue_band.shape[1]):
        plt.plot(phi, blue_band[:, i], c = 'mediumblue', ls = 'solid')

    plt.plot(phi, 0*phi, color = 'grey', ls = 'solid', lw = 1)

    plt.xticks(np.arange(0, 2*np.pi+1e-10, np.pi/2), ('0', r'$\pi/2$', r'$\pi$', r'$3\pi/2$', r'$2\pi$'))
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel(r'$\phi$')
    plt.ylabel('Energy (meV)')
    plt.title(r'$E_z$ = {} (meV)'.format(Ez))
    if savenm is not None:
        plt.savefig(savenm)
    plt.show()
