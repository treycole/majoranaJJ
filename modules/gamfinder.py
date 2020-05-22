import majoranaJJ.operators.sparse.qmsops as spop #sparse operators
import numpy as np
import scipy.linalg as LA
import scipy.sparse.linalg as spLA

def gamfinder(
    coor, ax, ay, NN, mu, NNb = None, Wj = 0, Sx = None, cutx = None, cuty = None, V = 0, gammax = 0, gammay = 0, gammaz = 0, alpha = 0, delta = 0 , phi = 0, qx = 0, qy = 0, periodicX = False, periodicY = False,
    k = 20, sigma = 0, which = 'LM', tol = 0, maxiter = None
    ):

    #saving the particle energies, all energies above E=0
    Ei = spop.EBDG(
        coor, ax, ay, NN, NNb = NNb, Wj = Wj,
        Sx = Sx, cutx = cutx, cuty = cuty,
        V = V, mu = mu,
        gammax = gammax, gammay = gammay, gammaz = gammaz,
        alpha = alpha, delta = delta, phi = phi,
        qx = qx, qy = qy,
        periodicX = periodicX, periodicY = periodicY,
        k = k, sigma = sigma, which = which, tol = tol, maxiter = maxiter
        )[int(k/2):][::2]
    #print(Ei)

    deltaG = 0.00001
    gammanew = gammax + deltaG

    #saving the particle energies, all energies above E=0
    Ef = spop.EBDG(
        coor, ax, ay, NN, NNb = NNb, Wj = Wj,
        Sx = Sx, cutx = cutx, cuty = cuty,
        V = V, mu = mu,
        gammax = gammanew, gammay = gammay, gammaz = gammaz,
        alpha = alpha, delta = delta, phi = phi,
        qx = qx, qy = qy,
        periodicX = periodicX, periodicY = periodicY,
        k = k, sigma = sigma, which = which, tol = tol, maxiter = maxiter
        )[int(k/2):][::2]
    #print(Ef)

    m = np.array((Ef - Ei)/(gammanew - gammax)) #slope, linear dependence on gamma
    #print(m)
    b = np.array(Ei - m*gammax) #y-intercept
    G_crit = np.array(-b/m) #gamma value that E=0 for given mu value
    #print(G_crit)

    return G_crit

"""
This function calculates the phase transition points. To work it needs energy eigenvalues and eigenvectors of unperturbed Hamiltonian, or a Hamiltonian without any Zeeman field. When a Zeeman field is turned on, perturbation theory can be applied to calculate the new energy eigenvalues and eigenvectors.
"""

def gamfinder_lowE(
    coor, ax, ay, NN, mu, NNb = None, Wj = 0, Sx = None, cutx = None, cuty = None, V = 0, gammax = 0, gammay = 0, gammaz = 0, alpha = 0, delta = 0 , phi = 0, qx = 0, qy = 0, periodicX = False, periodicY = False,
    k = 20, sigma = 0, which = 'LM', tol = 0, maxiter = None
    ):

    MU = mu #fixed mu value
    k0 = k #64 #perturbation energy eigs
    gx = np.linspace(0, 3, 5000) #just want two values that are close together in one step in the linspace

    H0 = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj, Sx=Sx, cutx=cutx, cuty=cuty, V=V, mu=MU, gammax=0.00001, alpha=alpha, delta=delta, phi=phi, qx=qx, qy=qy, periodicX=periodicX, periodicY=periodicY) #

    eigs_0, vecs_0 = spLA.eigsh(H0, k=k0, sigma=0, which='LM')
    vecs_0_hc = np.conjugate(np.transpose(vecs_0))

    H_G0 =  spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, Sx = Sx, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 0, alpha = alpha, delta = delta, phi = phi, qx = qx, qy=qy, periodicX = periodicX, periodicY=periodicY)

    H_G1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj, Sx = Sx, cutx = cutx, cuty = cuty, V = V, mu = MU, gammax = 1, alpha = alpha, delta = delta, phi = phi, qx = qx, qy=qy, periodicX = periodicX, periodicY=periodicY)

    #the proporitonality matrix for gammax, it is ones along the sites that have a gamma value

    HG = H_G1 - H_G0

    eig_arr = np.zeros((gx.shape[0], k))
    eig_arr_NB = np.zeros((gx.shape[0], k0))


    for i in range(2):

        H = H_G0 + gx[i]*HG #saves us from having to run matrix constructors in each loop. Still exact

        H_dif_basis = np.dot(vecs_0_hc, H.dot(vecs_0)) # H' = U^dagger H U
        eigs_dif_basis, U_dif_basis = LA.eigh(H_dif_basis)

        eig_arr_NB[i, :] = eigs_dif_basis


    Ei = eig_arr_NB[0, :][int(k/2):][::2]
    Ef = eig_arr_NB[1, :][int(k/2):][::2]

    gammax = gx[0]
    gammanew = gx[1]

    m = np.array((Ef - Ei)/(gammanew - gammax)) #slope, linear dependence on gamma
    b = np.array(Ei - m*gammax) #y-intercept
    G_crit = np.array(-b/m) #gamma value that E=0 for given mu value

    return G_crit
