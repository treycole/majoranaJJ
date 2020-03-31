import numpy as np

from majoranas import constants as const

#################### Descritizing kx operators #################################
"""
k-x operator
"""
def kx(coor, ax, ay, NN):

    N = coor.shape[0]
    k = np.zeros((N,N), dtype = "complex")

    for i in range(N):
        if NN[i,0] != -1:
            k[ NN[i,0] , i] = -1j/(2*ax)

        if NN[i, 2] != -1:
            k[ NN[i,2] , i] = 1j/(2*ax)

    return k

"""
k-x squared operator
"""
def kx2(coor, ax, ay, NN):

    N = coor.shape[0]
    k = np.zeros((N,N), dtype='complex')

    for i in range(N):
        k[i,i] = 2/ax**2

        if NN[i,0] != -1:
            k[ NN[i,0] , i] = -1/ax**2

        if NN[i, 2] != -1:
            k[ NN[i,0] , i] = -1/ax**2

    return k

#################### Descritizing ky operators #################################
"""
k-y operator
"""
def ky(coor, ax, ay, NN):

    N = coor.shape[0]
    k = np.zeros((N,N), dtype = "complex")

    for i in range(N):
        if NN[i,1] != -1:
            k[ NN[i,1] , i] = -1j/(2*ay)

        if NN[i, 3] != -1:
            k[ NN[i,3] , i] = 1j/(2*ay)

    return k

"""
k-y squared operator
"""
def ky2(coor, ax, ay, NN):

    N = coor.shape[0]
    k = np.zeros((N,N), dtype='complex')

    for i in range(N):
        k[i,i] = 2/ay**2

        if NN[i,1] != -1:
            k[ NN[i,1] , i] = -1/ay**2

        if NN[i, 3] != -1:
            k[ NN[i,3] , i] = -1/ay**2

    return k


#################### Periodic kx operators ##################################

def kpx(coor, ax, ay, NN, NNb, qx = 0):
    N = coor.shape[0]   #Number of Lattice sites
    xmin = min(coor[:, 0])
    xmax = max(coor[:, 0])

    Lx = (xmax - xmin + 1)*ax

    k = np.zeros((N,N), dtype = "complex")

    for i in range(N):
        if NN[i,0] != -1:
            k[ NN[i,0] , i] = -1j/(2*ax)

        if NN[i, 2] != -1:
            k[ NN[i,0] , i] = 1j/(2*ax)

        if NNb[i, 0] != -1:
            k[ NNb[i,0] , i] = ( -1j/2*ax )*np.exp( -1j*qx*Lx )

        if NNb[i, 2] != -1:
            k[ NNb[i,2] , i] = ( 1j/2*ax )*np.exp( 1j*qx*Lx )

    return k

def kpx2(coor, ax, ay, NN, NNb, qx = 0):
    N = coor.shape[0]
    xmin = min(coor[:, 0])
    xmax = max(coor[:, 0])

    Lx = (xmax - xmin + 1)*ax

    k = np.zeros((N,N), dtype = "complex")

    for i in range(N):
        k[i,i] = 2/ax**2

        if NN[i,0] != -1:
            k[ NN[i, 0] , i] = -1/ax**2

        if NN[i, 2] != -1:
            k[ NN[i, 2] , i] = -1/ax**2

        if NNb[i, 0] != -1:
            k[ NNb[i,0] , i] = ( -1/ax**2 )*np.exp( -1j*qx*Lx )

        if NNb[i, 2] != -1:
            k[ NNb[i, 2] , i] = ( -1/ax**2 )*np.exp( 1j*qx*Lx )

return k

#################### Periodic ky operators ##################################

def kpy(coor, ax, ay, NN, NNb, qy = 0):
    N = coor.shape[0]
    ymin = min(coor[:, 1])
    ymax = max(coor[:, 1])

    Ly = (ymax - ymin + 1)*ay

    k = np.zeros((N,N), dtype = "complex")

    for i in range(N):
        if NN[i,1] != -1:
            k[NN[i,1],i] = -1j/(2*ax)

        if NN[i, 3] != -1:
            k[NN[i,3],i] = 1j/(2*ax)

        if NNb[i, 1] != -1:
            k[NNb[i,1], i] = ( -1j/2*ax )*np.exp( -1j*qx*Ly )
            #Hopping to next unit cell, e^ikL

        if NNb[i, 3] != -1:
            k[ NNb[i,3] , i ] = ( 1j/2*ax )*np.exp( 1j*qx*Ly )

    return k

def kpy2(coor, ax, ay, NN, NNb, qy = 0):
    N = coor.shape[0]
    ymin = min(coor[:, 1])
    ymax = max(coor[:, 1])

    Ly = (ymax - ymin + 1)*ay

    k = np.zeros((N,N), dtype='complex')

    for i in range(N):
        k[i,i] = 2/ay**2

        if NN[i,1] != -1:
            k[ NN[i, 1] , i] = -1/ay**2

        if NN[i, 3] != -1:
            k[ NN[i, 3] , i ] = -1/ay**2

        if NNb[i, 1] != -1:
            k[ NNb[i,1] , i ] = ( -1/ay**2 )*np.exp( -1j*qy*Ly )

        if NNb[i, 3] != -1:
            k[ NNb[i, 3] , i ] = ( -1/ay**2 )*np.exp( 1j*qy*Ly )
    return k

###################### Delta Matrix ###############################

def Delta(coor, delta, Wsc, Wj, phi = 0, Sx = 0, Sy = 0, cutx = 0, cuty = 0):
    N = coor.shape[0]
    D = np.zeros((N, N), dtype = 'complex')

    for i in range(N):
        y = coor[i,1]
        x = coor[i,0]

        if y <= Wsc:
            D[i,i] = delta*np.exp( -1j*phi/2 )

        if y > Wsc and y <= (Wsc+Wj):
            D[i,i] = 0

        if y > (Wsc+Wj):
            D[i,i] = delta*np.exp( 1j*phi/2 )

    #Delta = delta*np.eye(N, N, dtype = 'complex')
    D00 = np.zeros((N,N))
    D11 = np.zeros((N,N))
    D01 = D
    D10 = -D
    D = np.block([[D00, D01], [D10, D11]])
    return D

###################### Hamiltonians ################################

"""
This is the Hamiltonian with Spin Orbit Coupling and nearest neighbor h
opping and no Superconductivity.

The parameter PERIODIC determines whether the function calls a construction of
k-operators with or without boundary conditions in x and y directions.

Basis: Two states per lattice site for spin up and down. Rows/Columns 1 ->
N correspond to spin up, rows/columns n -> 2N correspond to spin down
"""

def H0(
    coor, ax, ay, NN, NNb = None,
    potential = 0,
    mu = 0,
    gammax = 0, gammay = 0, gammaz = 0,
    alpha = 0,
    qx = 0, qy = 0,
    periodicx = 'no', periodicy = 'no'
    ):

    if periodicx.lower() == 'yes':
        if NNb == None:
            print("If periodic need boundary array")
            return
        k_x = kpx(coor, ax, ay, NN, NNb, qx = qx)
        k_x2 = kpx2(coor, ax, ay, NN, NNb, qx = qx)

    if periodicx.lower() == 'no':
        if NN == None:
            print("Need neighbor array")
            return
        k_x = kx(coor, ax, ay, NN)
        k_x2 = kx2(coor, ax, ay, NN)

    if periodicy.lower() == 'yes':
        if NNb == None:
            print("If periodic need boundary array")
            return
        k_y = kpy(coor, ax, ay, NN, NNb, qy = qy)
        k_y2 = kpy2(coor, ax, ay, NN, NNb, qy = qy)

    if periodicy.lower() == 'no':
        if NN == None:
            print("Need neighbor array")
            return
        k_y = ky(coor, ax, ay, NN)
        k_y2 = ky2(coor, ax, ay, NN)

    N = coor.shape[0]
    I = np.eye(N,N) #Identity Matrix

    #Hfree = np.zeros((N,N), dtype = 'complex')
    Hfree = (const.xi/2)*(k_x2 + k_y2)
    V = potential
    MU = mu*I

    H00 = Hfree + gammaz*I + V - MU
    H11 = Hfree - gammaz*I + V - MU
    H10 = alpha*(1j*k_x - k_y) + gammax*I + 1j*gammay*I
    H01 = alpha*(-1j*k_x - k_y) + gammax*I - 1j*gammay*I

    H = np.block([[H00, H01], [H10, H11]])
    return H

"""
This is the Bogoliobuv de Gennes Hamiltonian with nearest neighbor hopping,
Spin Orbit Coupling, Superconductivity, and Zeeman energy contributions.

The parameters Sx, Sy, cutx, and cuty determine the geometry of the Josephson
Junction along the boundary between the superconductor and 2DEG.
"""
#TODO need to implement cut conditions

def HBDG(
    coor, ax, ay, NN, Wsc, Wj, NNb = None,
    potential = 0,
    mu = 0,
    gammax = 0, gammay = 0, gammaz = 0,
    delta = 0, phi = 0, Sx = 0, Sy = 0, cutx = 0, cuty =0,
    alpha = 0,
    qx = 0, qy = 0,
    periodicx = 'no', periodicy = 'no'
    ):

    D = Delta(coor, delta, Wsc, Wj, phi = phi, Sx = Sx, Sy = Sy, cutx = cutx, cuty = cuty)

    H00 = H0(coor, ax, ay, NN, NNb = NNb, potential = potential, mu = mu, gammax = gammax, gammay = gammay, gammaz = gammaz,
        alpha = alpha, qx = qx, qy = qy, periodicx = periodicx, periodicy = periodicy)
    H01 = D
    H10 = -np.conjugate(D)
    H11 = -np.conjugate( H0(coor, ax, ay, NN, NNb = NNb, potential = potential, mu = mu, gammax = gammax, gammay = gammay,
        gammaz = gammaz, alpha = alpha, qx = -qx, qy = -qy, periodicx = periodicx, periodicy = periodicy) )

    HBDG = np.block([[H00, H01] , [H10, H11]])
    return HBDG

######################## Potential shapes ##############################

def V_barrier(V0, xi, xf, coor):
#(Amplitude, starting point of barrier, ending pt of barrier, coordinate array)
    N = coor.shape[0]
    V = np.zeros((N, N))
    for i in range(N):
        if coor[i,0] < xf and coor[i,0] > xi:
            V[i,i] = V0
    return V

def V_periodic(V0, coor):
    N = coor.shape[0]
    V = np.zeros((N,N))
    Lx = (max(coor[:, 0]) - min(coor[:, 0]))
    #Unit cell size in x-direction, no +1 because has to match coor array period
    Ly = (max(coor[:, 1]) - min(coor[:, 1])) #Unit cell size in y-direction
    for i in range(N):
        V[i,i] = V0 * np.sin( np.pi*(coor[i,0])/Lx ) * np.sin( np.pi*coor[i,1]/Ly )
    return V
