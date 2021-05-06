import numpy as np
import scipy.sparse as sparse
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import majoranaJJ.modules.checkers as check

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

def delta_profile(coor, delta):
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

    plt.show()

def potential_profile(coor, V):
    V = V.toarray()
    N = coor.shape[0]
    fig = plt.figure()
    axx = fig.add_subplot(1,1,1)
    for i in range(N):
        if V[i, i] < 0:
            axx.scatter(coor[i, 0], coor[i, 1], c = 'b')
        if V[i, i] > 0:
            axx.scatter(coor[i, 0], coor[i, 1], c = 'red')
        if V[i, i]==0:
            axx.scatter(coor[i, 0], coor[i, 1], c = 'k')
    blue_patch = mpatches.Patch(color='b', label='Negative potential')
    red_patch = mpatches.Patch(color='r', label='Positive potential')
    plt.legend(handles=[blue_patch,red_patch])
    axx.set_xlim(-5, max(coor[:,0])+5)
    axx.set_title('Potential Profile', wrap = True)
    axx.set_aspect(1.0)
    plt.show()

def Zeeman_profile(coor, V):
    V = V.toarray()
    N = coor.shape[0]
    fig = plt.figure()
    axx = fig.add_subplot(1,1,1)
    for i in range(N):
        if V[i, i] != 0:
            axx.scatter(coor[i, 0], coor[i, 1], c = 'b')
        if V[i, i]==0:
            axx.scatter(coor[i, 0], coor[i, 1], c = 'k')
    blue_patch = mpatches.Patch(color='b', label='Zeeman != 0')
    black_patch = mpatches.Patch(color='k', label='Zeeman == 0 ')
    plt.legend(handles=[blue_patch,black_patch])
    axx.set_xlim(-5, max(coor[:,0])+5)
    axx.set_title('Zeeman Profile', wrap = True)
    axx.set_aspect(1.0)
    plt.show()

"""
Plotting the probability density
"""
def probdens_cmap(
    coor, Wj, cutxT, cutxB, cutyT, cutyB, eigs, states,
    n = 1, cmap = 'hot', savenm = None
    ):

    N = coor.shape[0]
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
    Wsc = int((Ny - Wj)/2) #width of single superconductor

    num_div = int(states.shape[0]/N)
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    states = states[:, idx_sort]

    probdens = np.square(abs(states[:, n]))
    map = np.zeros(N)
    for i in range(num_div):
        map[:] = map[:] + probdens[i*N : (i+1)*N]

    print("Sum of prob density", sum(map))
    if Nx > 3:
        fig = plt.figure()
        axx = fig.add_subplot(1,1,1)
        tc = axx.tricontourf(coor[:,0], coor[:,1], map, 1000)
        top_linex = []
        top_liney = []
        bottom_linex = []
        bottom_liney = []
        for i in range(N-Nx):
            bool_inSC, which = check.is_in_SC(i, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
            bool_inSCnbr, whichnbr = check.is_in_SC(i+Nx, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
            if bool_inSC and not bool_inSCnbr and which == 'B':
                bottom_linex.append(coor[i,0])
                bottom_liney.append(coor[i,1])
            if not bool_inSC and bool_inSCnbr and whichnbr == 'T':
                top_linex.append(coor[i,0])
                top_liney.append(coor[i,1]+1)


        top_linex = np.array(top_linex)
        top_liney = np.array(top_liney)
        bottom_linex = np.array(bottom_linex)
        bottom_liney = np.array(bottom_liney)
        idx_top = np.argsort(top_linex)
        idx_bottom = np.argsort(bottom_linex)
        #print(idx_top)
        top_linex = top_linex[idx_top]
        top_liney = top_liney[idx_top]
        bottom_linex = bottom_linex[idx_bottom]
        bottom_liney = bottom_liney[idx_bottom]
        #print(top_linex, top_liney)

        #print(len(top_liney))
        top_lineytmp = []
        top_linextmp = []
        for i in range(top_liney.shape[0]-1):
            #print(top_liney[i], top_liney[i+1])
            if top_liney[i] != top_liney[i+1]:
                #print('here')
                top_linextmp.append(top_linex[i+1])
                top_lineytmp.append(top_liney[i]+1)

        top_linex = np.concatenate((top_linex,np.array(top_linextmp)), axis=None)
        top_liney = np.concatenate((top_liney,np.array(top_lineytmp)), axis=None)
        axx.scatter(bottom_linex, bottom_liney, c = 'r')
        axx.scatter(top_linex, top_liney, c = 'r')
        fig.colorbar(tc)
        title = r'$|\psi|^2$'
        axx.set_title(title)
        if savenm is not None:
            plt.savefig(savenm)
        print("Energy Value of State", eigs[n])
        plt.show()

    else:
        x_idx = 1
        oneD = np.zeros(Ny)
        for j in range(Ny):
            oneD[j] = map[x_idx+j*Nx]
        plt.plot(np.linspace(0, Ny-1, Ny), oneD, c='b')
        plt.vlines(Ny-Wsc, -.001, max(oneD), color='g')
        plt.vlines(0+Wsc-1, -.001, max(oneD), color='g')
        plt.grid()
        plt.ylabel(r'$|\psi|^2$')
        plt.ylabel('y')
        plt.show()

def state_cmap(
    coor, Wj, cutxT, cutxB, cutyT, cutyB, eigs, states,
    n = 1, cmap = 'hot',
    savenm = None
    ):

    N = coor.shape[0]
    Nx = int((max(coor[: , 0]) - min(coor[:, 0])) + 1) #number of lattice sites in x-direction, parallel to junction
    Ny = int((max(coor[: , 1]) - min(coor[:, 1])) + 1) #number of lattice sites in y-direction, perpendicular to junction
    Wsc = int((Ny - Wj)/2) #width of single superconductor
    num_div = int(states.shape[0]/N)
    idx_sort = np.argsort(eigs)
    eigs = eigs[idx_sort]
    states = states[:, idx_sort]
    print("Energy Value of State", eigs[n])
    mapRe_PU = np.zeros(N)
    mapRe_PD = np.zeros(N)
    mapRe_HU = np.zeros(N)
    mapRe_HD = np.zeros(N)
    mapIm_PU  = np.zeros(N)
    mapIm_PD  = np.zeros(N)
    mapIm_HU  = np.zeros(N)
    mapIm_HD  = np.zeros(N)

    i=0
    mapRe_PU[:] = states[i*N : (i+1)*N, n]
    mapIm_PU[:] = np.imag(states[i*N : (i+1)*N, n])
    i=1
    mapRe_PD[:] = states[i*N : (i+1)*N, n]
    mapIm_PD[:] = np.imag(states[i*N : (i+1)*N, n])
    i=2
    mapRe_HU[:] = states[i*N : (i+1)*N, n]
    mapIm_HU[:] = np.imag(states[i*N : (i+1)*N, n])
    i=3
    mapRe_HD[:] = states[i*N : (i+1)*N, n]
    mapIm_HD[:] = np.imag(states[i*N : (i+1)*N, n])

    top_linex = []
    top_liney = []
    bottom_linex = []
    bottom_liney = []
    for i in range(N-Nx):
        bool_inSC, which = check.is_in_SC(i, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
        bool_inSCnbr, whichnbr = check.is_in_SC(i+Nx, coor, Wsc, Wj, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB)
        if bool_inSC and not bool_inSCnbr and which == 'B':
            bottom_linex.append(coor[i,0])
            bottom_liney.append(coor[i,1])
        if not bool_inSC and bool_inSCnbr and whichnbr == 'T':
            top_linex.append(coor[i,0])
            top_liney.append(coor[i,1]+1)
    top_linex = np.array(top_linex)
    top_liney = np.array(top_liney)
    bottom_linex = np.array(bottom_linex)
    bottom_liney = np.array(bottom_liney)
    idx_top = np.argsort(top_linex)
    idx_bottom = np.argsort(bottom_linex)
    top_linex = top_linex[idx_top]
    top_liney = top_liney[idx_top]
    bottom_linex = bottom_linex[idx_bottom]
    bottom_liney = bottom_liney[idx_bottom]
    #print(idx_top)
    #print(top_linex, top_liney)
    #print(len(top_liney))
    top_lineytmp = []
    top_linextmp = []
    for i in range(top_liney.shape[0]-1):
        #print(top_liney[i], top_liney[i+1])
        if top_liney[i] != top_liney[i+1]:
            #print('here')
            top_linextmp.append(top_linex[i+1])
            top_lineytmp.append(top_liney[i]+1)
    top_linex = np.concatenate((top_linex,np.array(top_linextmp)), axis=None)
    top_liney = np.concatenate((top_liney,np.array(top_lineytmp)), axis=None)

    if Nx > 3:
        fig = plt.figure()
        axx = fig.add_subplot(1,1,1)
        axx.scatter(bottom_linex, bottom_liney, c = 'r')
        axx.scatter(top_linex, top_liney, c = 'r')
        tc = axx.tricontourf(coor[:,0], coor[:,1], mapRe, 1000)
        fig.colorbar(tc)
        axx.set_title(r'Re($\psi$)')
        plt.show()

        fig = plt.figure()
        axx = fig.add_subplot(1,1,1)
        axx.scatter(bottom_linex, bottom_liney, c = 'r')
        axx.scatter(top_linex, top_liney, c = 'r')
        tc = axx.tricontourf(coor[:,0], coor[:,1], mapIm, 1000)
        fig.colorbar(tc)
        axx.set_title(r'Im($\psi$)')
        plt.show()
    else:
        x_idx = 1
        oneD_Re_PU = np.zeros(Ny)
        oneD_Im_PU = np.zeros(Ny)
        oneD_Re_PD = np.zeros(Ny)
        oneD_Im_PD = np.zeros(Ny)
        oneD_Re_HU = np.zeros(Ny)
        oneD_Im_HU = np.zeros(Ny)
        oneD_Re_HD = np.zeros(Ny)
        oneD_Im_HD = np.zeros(Ny)
        for j in range(Ny):
            oneD_Re_PU[j] = mapRe_PU[x_idx+j*Nx]
            oneD_Im_PU[j] = mapIm_PU[x_idx+j*Nx]
            oneD_Re_PD[j] = mapRe_PD[x_idx+j*Nx]
            oneD_Im_PD[j] = mapIm_PD[x_idx+j*Nx]
            oneD_Re_HU[j] = mapRe_HU[x_idx+j*Nx]
            oneD_Im_HU[j] = mapIm_HU[x_idx+j*Nx]
            oneD_Re_HD[j] = mapRe_HD[x_idx+j*Nx]
            oneD_Im_HD[j] = mapIm_HD[x_idx+j*Nx]
        plt.plot(np.linspace(0, Ny-1, Ny), oneD_Re_PU, label='real', c='b')
        plt.plot(np.linspace(0, Ny-1, Ny), oneD_Im_PU, label='imaginary', c='r')
        plt.vlines(Ny-Wsc, -.05, 0.11, color='g')
        plt.vlines(0+Wsc-1, -.05, 0.11, color='g')
        plt.ylabel(r'$\psi(y) Particle Up$')
        plt.xlabel('y')
        plt.legend()
        plt.show()
        plt.plot(np.linspace(0, Ny-1, Ny), oneD_Re_PD, label='real', c='b')
        plt.plot(np.linspace(0, Ny-1, Ny), oneD_Im_PD, label='imaginary', c='r')
        plt.vlines(Ny-Wsc, -.05, 0.11, color='g')
        plt.vlines(0+Wsc-1, -.05, 0.11, color='g')
        plt.ylabel(r'$\psi(y) Particle Down$')
        plt.xlabel('y')
        plt.legend()
        plt.show()
        plt.plot(np.linspace(0, Ny-1, Ny), oneD_Re_HU, label='real', c='b')
        plt.plot(np.linspace(0, Ny-1, Ny), oneD_Im_HU, label='imaginary', c='r')
        plt.vlines(Ny-Wsc, -.05, 0.11, color='g')
        plt.vlines(0+Wsc-1, -.05, 0.11, color='g')
        plt.ylabel(r'$\psi(y) Hole Up$')
        plt.xlabel('y')
        plt.legend()
        plt.show()
        plt.plot(np.linspace(0, Ny-1, Ny), oneD_Re_HD, label='real', c='b')
        plt.plot(np.linspace(0, Ny-1, Ny), oneD_Im_HD, label='imaginary', c='r')
        plt.vlines(Ny-Wsc, -.05, 0.11, color='g')
        plt.vlines(0+Wsc-1, -.05, 0.11, color='g')
        plt.ylabel(r'$\psi(y) Hole Down$')
        plt.xlabel('y')
        plt.legend()
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
        plt.subplots_adjust(top=0.85)
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
