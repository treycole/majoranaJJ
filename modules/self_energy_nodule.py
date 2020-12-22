import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as Spar
import scipy.sparse.linalg as spLA
from scipy.signal import argrelextrema

import majoranaJJ.modules.parameters as par
import majoranaJJ.modules.constants as const
import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes

def top_SC_sNRG_calc(omega, Wj, nodx, ax, ay_targ, kx, m_eff, alp_l, alp_t, mu, Gam_SC, delta, phi, iter, eta):
    # Calculates the bulk and surface Greens functions of the top superconductor
    # along with the self-energy that it produces in the junction region
    #    * omega is the (real) energy that goes into the Greens function
    #    * W is the width of the junction
    #    * ay_targ is the targeted lattice constant
    #    * kx is the wavevector along the length of the junction
    #    * m_eff is the effective mass
    #    * alp_l is the longitudinal spin-orbit coupling coefficient
    #    * alp_t is the transverse spin-orbit coupling coefficient
    #    * mu is the chemical potential
    #    * delta is the (complex) SC order parameter in the top SC
    #    * iter is the number of iteration of the algorithm to perform
    #    * eta is the imaginary component of the energy that is used for broadening

    if nodx == 0:
        Nx = 3
    else:
        Nx = nodx+2

    Ny = int(Wj/ay_targ) # number of lattice sites in the junction region (in the y-direction)
    #ay = Wj/float(Ny+1)      # actual lattice constant
    ay = ay_targ
    Ny = Ny+2 #add one SC site on each side
    Wj_int = Ny - 2

    tx = -1000*par.hbm0/(2*m_eff*ax**2)  # spin-preserving hopping strength
    ty = -1000*par.hbm0/(2*m_eff*ay**2) # spin-preserving hopping strength
    ty_alp = alp_t/(2*ay) # spin-orbit hopping strength in the y-direction
    tx_alp = alp_l/(2*ax)  # onsite spin-orbit coupling contribution
    ep_on = -2*tx - 2*ty - mu
    dc = np.conjugate(delta)

    ### Onsite Hamiltonian matrix
    H00 = np.zeros((4*Nx, 4*Nx), dtype='complex')
    H10 = np.zeros((4*Nx, 4*Nx), dtype='complex')
    H01 = np.zeros((4*Nx, 4*Nx), dtype='complex')

    for i in range(Nx):
        #particle
        H10[i, i] = ty
        H10[i+Nx, i+Nx] = ty
        #hole
        H10[i+2*Nx, i+2*Nx] = -ty
        H10[i+3*Nx, i+3*Nx] = -ty
        #particle
        H10[i, i+Nx] = 1j*ty_alp #check sign
        H10[i+Nx, i] = 1j*ty_alp
        #hole
        H10[i+2*Nx, i+Nx+2*Nx] = 1j*ty_alp
        H10[i+Nx+2*Nx, i+2*Nx] = 1j*ty_alp

        #particle
        H00[i, i] = ep_on
        H00[i+Nx, i+Nx] = ep_on
        #hole
        H00[i+2*Nx, i+2*Nx] = -ep_on
        H00[i+3*Nx, i+3*Nx] = -ep_on

        H00[i, i+Nx+2*Nx] = delta
        H00[i+Nx, i+2*Nx] = -delta
        H00[i+2*Nx, i+Nx] = -dc
        H00[i+Nx+2*Nx, i] = dc
        if i != Nx-1:
            H00[i, i+1] = tx
            H00[i+1, i] = tx
            H00[i+Nx, i+Nx+1] = tx
            H00[i+Nx+1, i+Nx] = tx

            H00[i+2*Nx, i+2*Nx+1] = -tx
            H00[i+2*Nx+1, i+2*Nx] = -tx
            H00[i+3*Nx, i+3*Nx+1] = -tx
            H00[i+3*Nx+1, i+3*Nx] = -tx

            H00[i+1, i+Nx] = -tx_alp
            H00[i+Nx, i+1] = -tx_alp
            H00[i, i+Nx+1] = tx_alp
            H00[i+Nx+1, i] = tx_alp

            H00[i+1+2*Nx, i+Nx+2*Nx] = tx_alp
            H00[i+Nx+2*Nx, i+1+2*Nx] = tx_alp
            H00[i+2*Nx, i+Nx+1+2*Nx] = -tx_alp
            H00[i+Nx+1+2*Nx, i+2*Nx] = -tx_alp

        if i == Nx-1:
            #particle
            H00[i, 0] += tx*np.exp(1j*kx*ax)
            H00[0, i] += tx*np.exp(-1j*kx*ax)
            H00[i+Nx, 0+Nx] += tx*np.exp(1j*kx*ax)
            H00[0+Nx, i+Nx] += tx*np.exp(-1j*kx*ax)

            H00[i, 0+Nx] += -tx_alp*np.exp(1j*kx*ax)
            H00[0+Nx, i] += -tx_alp*np.exp(-1j*kx*ax)
            H00[i+Nx, 0 ] += tx_alp*np.exp(1j*kx*ax)
            H00[0, i+Nx] += tx_alp*np.exp(-1j*kx*ax)

            #hole
            H00[i+2*Nx, 0+2*Nx] += np.conjugate(-tx*np.exp(-1j*kx*ax))
            H00[0+2*Nx, i+2*Nx] += np.conjugate(-tx*np.exp(1j*kx*ax))
            H00[i+Nx+2*Nx, 0+Nx+2*Nx] += np.conjugate(-tx*np.exp(-1j*kx*ax))
            H00[0+Nx+2*Nx, i+Nx+2*Nx] += np.conjugate(-tx*np.exp(1j*kx*ax))

            H00[i+2*Nx, 0+Nx+2*Nx] += np.conjugate(tx_alp*np.exp(-1j*kx*ax))
            H00[0+Nx+2*Nx, i+2*Nx] += np.conjugate(tx_alp*np.exp(1j*kx*ax))
            H00[i+Nx+2*Nx, 0+2*Nx ] += np.conjugate(-tx_alp*np.exp(-1j*kx*ax))
            H00[0+2*Nx, i+Nx+2*Nx] += np.conjugate(-tx_alp*np.exp(1j*kx*ax))

    print(H00)
    #print(H10.real)
    #print()
    #print(H10.imag)
    ### Identity matrix
    I = np.eye(4*Nx, dtype = 'complex')
    H01 = np.conjugate(np.transpose(H10))
    ### Performing iterative algorithm
    E = omega + 1j*eta
    ep_o = H00
    eps_o = H00
    alp_o = H01
    beta_o = H10
    for i in range(iter):
        g_o = np.linalg.inv( E*I - ep_o )
        alp_n = np.dot(alp_o, np.dot(g_o,alp_o))
        beta_n = np.dot(beta_o, np.dot(g_o,beta_o))
        ep_n = ep_o + np.dot(alp_o, np.dot(g_o,beta_o)) + np.dot(beta_o, np.dot(g_o,alp_o))
        eps_n = eps_o + np.dot(alp_o, np.dot(g_o,beta_o))

        ep_o = 1.*ep_n
        eps_o = 1.*eps_n
        alp_o = 1.*alp_n
        beta_o = 1.*beta_n

    G_s = np.linalg.inv(E*I - eps_o)
    G_b = np.linalg.inv(E*I - ep_o)

    sNRG = np.dot(H01,np.dot(G_s,H10))

    row = []; col = []; data = []
    N = Nx*Ny
    for m in range(4):
        for n in range(4):
            for i in range(Nx):
                for j in range(Nx):
                    row.append((Ny-1)*Nx + i + m*Nx*Ny); col.append((Ny-1)*Nx + j + n*Nx*Ny); data.append(sNRG[m*Nx + i, n*Nx + j])
    sNRG_mtx = Spar.csc_matrix((data,(row,col)), shape = (4*N,4*N), dtype = 'complex')

    return G_s, G_b, sNRG_mtx

def bot_SC_sNRG_calc(omega, Wj, nodx, ax, ay_targ, kx, m_eff, alp_l, alp_t, mu, Gam_SC, delta, phi, iter, eta):
    # Calculates the bulk and surface Greens functions of the bottom superconductor
    # along with the self-energy that it produces in the junction region
    #     * omega is the (real) energy that goes into the Greens function
    #     * W is the width of the junction
    #     * ay_targ is the targeted lattice constant
    #     * kx is the wavevector along the length of the junction
    #     * m_eff is the effective mass
    #     * alp_l is the longitudinal spin-orbit coupling coefficient
    #     * alp_t is the transverse spin-orbit coupling coefficient
    #     * mu is the chemical potential
    #     * delta is the (complex) SC order parameter in the top SC
    #     * iter is the number of iteration of the algorithm to perform
    #     * eta is the imaginary component of the energy that is used for broadening

    if nodx == 0:
        Nx = 3
    else:
        Nx = nodx+2

    Ny = int(Wj/ay_targ) # number of lattice sites in the junction region (in the y-direction)
    #ay = Wj/float(Ny+1)      # actual lattice constant
    ay = ay_targ
    Ny = Ny+2 #add one SC site on each side
    Wj_int = Ny - 2

    tx = -1000.*par.hbm0/(2.*m_eff*ax**2)  # spin-preserving hopping strength
    ty = -1000.*par.hbm0/(2.*m_eff*ay**2) # spin-preserving hopping strength
    ty_alp = alp_t /(2*ay) # spin-orbit hopping strength in the y-direction
    tx_alp = alp_l/(2*ax)  # onsite spin-orbit coupling contribution
    ep_on = -2*tx - 2*ty - mu
    #ep_on = -2*ty - mu
    delta = delta*np.exp(1j*phi)
    dc = np.conjugate(delta)

    ### Onsite Hamiltonian matrix
    H00 = np.zeros((4*Nx, 4*Nx), dtype='complex')
    H10 = np.zeros((4*Nx, 4*Nx), dtype='complex')
    H01 = np.zeros((4*Nx, 4*Nx), dtype='complex')

    for i in range(Nx):
        #particle
        H10[i, i] = ty
        H10[i+Nx, i+Nx] = ty
        #hole
        H10[i+2*Nx, i+2*Nx] = -ty #-H0(-k)*
        H10[i+3*Nx, i+3*Nx] = -ty #-H0(-k)*
        #particle
        H10[i, i+Nx] = -1j*ty_alp #bottom SC, negative sign
        H10[i+Nx, i] = -1j*ty_alp #bottom SC, negative sign
        #hole
        H10[i+2*Nx, i+Nx+2*Nx] = -1j*ty_alp #bottom SC, negative sign
        H10[i+Nx+2*Nx, i+2*Nx] = -1j*ty_alp #bottom SC, negative sign

        #particle
        H00[i, i] = ep_on
        H00[i+Nx, i+Nx] = ep_on
        #hole
        H00[i+2*Nx, i+2*Nx] = -ep_on
        H00[i+3*Nx, i+3*Nx] = -ep_on

        H00[i, i+Nx+2*Nx] = delta #delta
        H00[i+Nx, i+2*Nx] = -delta
        H00[i+2*Nx, i+Nx] = -dc
        H00[i+Nx+2*Nx, i] = dc
        if i != Nx-1:
            #particle
            H00[i, i+1] = tx
            H00[i+1, i] = tx
            H00[i+Nx, i+Nx+1] = tx
            H00[i+Nx+1, i+Nx] = tx

            #hole
            H00[i+2*Nx, i+2*Nx+1] = -tx
            H00[i+2*Nx+1, i+2*Nx] = -tx
            H00[i+3*Nx, i+3*Nx+1] = -tx
            H00[i+3*Nx+1, i+3*Nx] = -tx

            #particle
            H00[i+1, i+Nx] = -tx_alp
            H00[i+Nx, i+1] = -tx_alp
            H00[i, i+Nx+1] = tx_alp
            H00[i+Nx+1, i] = tx_alp

            #hole
            H00[i+1+2*Nx, i+Nx+2*Nx] = tx_alp
            H00[i+Nx+2*Nx, i+1+2*Nx] = tx_alp
            H00[i+2*Nx, i+Nx+1+2*Nx] = -tx_alp
            H00[i+Nx+1+2*Nx, i+2*Nx] = -tx_alp

        if i == Nx-1:
            #particle
            H00[i, 0] += tx*np.exp(1j*kx*ax)
            H00[0, i] += tx*np.exp(-1j*kx*ax)
            H00[i+Nx, 0+Nx] += tx*np.exp(1j*kx*ax)
            H00[0+Nx, i+Nx] += tx*np.exp(-1j*kx*ax)

            H00[i, 0+Nx] += -tx_alp*np.exp(1j*kx*ax)
            H00[0+Nx, i] += -tx_alp*np.exp(-1j*kx*ax)
            H00[i+Nx, 0 ] += tx_alp*np.exp(1j*kx*ax)
            H00[0, i+Nx] += tx_alp*np.exp(-1j*kx*ax)

            #hole
            H00[i+2*Nx, 0+2*Nx] += np.conjugate(-tx*np.exp(-1j*kx*ax))
            H00[0+2*Nx, i+2*Nx] += np.conjugate(-tx*np.exp(1j*kx*ax))
            H00[i+Nx+2*Nx, 0+Nx+2*Nx] += np.conjugate(-tx*np.exp(-1j*kx*ax))
            H00[0+Nx+2*Nx, i+Nx+2*Nx] += np.conjugate(-tx*np.exp(1j*kx*ax))

            H00[i+2*Nx, 0+Nx+2*Nx] += np.conjugate(tx_alp*np.exp(-1j*kx*ax))
            H00[0+Nx+2*Nx, i+2*Nx] += np.conjugate(tx_alp*np.exp(1j*kx*ax))
            H00[i+Nx+2*Nx, 0+2*Nx] += np.conjugate(-tx_alp*np.exp(-1j*kx*ax))
            H00[0+2*Nx, i+Nx+2*Nx] += np.conjugate(-tx_alp*np.exp(1j*kx*ax))

    ### Identity matrix
    I = np.eye(4*Nx, dtype = 'complex')
    H01 = np.conjugate(np.transpose(H10))
    ### Performing iterative algorithm
    E = omega + 1j*eta
    ep_o = H00
    eps_o = H00
    alp_o = H01
    beta_o = H10
    for i in range(iter):
        g_o = np.linalg.inv( E*I - ep_o )
        alp_n = np.dot(alp_o, np.dot(g_o,alp_o))
        beta_n = np.dot(beta_o, np.dot(g_o,beta_o))
        ep_n = ep_o + np.dot(alp_o, np.dot(g_o,beta_o)) + np.dot(beta_o, np.dot(g_o,alp_o))
        eps_n = eps_o + np.dot(alp_o, np.dot(g_o,beta_o))

        ep_o = 1.*ep_n
        eps_o = 1.*eps_n
        alp_o = 1.*alp_n
        beta_o = 1.*beta_n

    G_s = np.linalg.inv(E*I - eps_o)
    G_b = np.linalg.inv(E*I - ep_o)

    sNRG = np.dot(H01,np.dot(G_s,H10))

    row = []; col = []; data = []
    N = Nx*Ny
    for m in range(4):
        for n in range(4):
            for i in range(Nx):
                for j in range(Nx):
                    #print(m,n,i)
                    row.append(i + m*Nx*Ny); col.append(j + n*Nx*Ny); data.append(sNRG[m*Nx + i, n*Nx + j])
    sNRG_mtx = Spar.csc_matrix((data,(row,col)), shape = (4*N, 4*N), dtype = 'complex')

    return G_s, G_b, sNRG_mtx

def Junc_eff_Ham_gen(omega, Wj, nodx, nody, ax, ay_targ, kx, m_eff, alp_l, alp_t, mu, Vj, Gam, delta, phi, Gam_SC_factor=0, iter=50, eta=0):
    ### Generates the effective Hamiltonian for the Junction, which includes the self-energy from both of the SC regions
    ###     * omega is the (real) energy that goes into the Green's function
    ###     * W is the width of the junction
    ###     * ay_targ is the targeted lattice constant
    ###     * kx is the wavevector along the length of the junction
    ###     * m_eff is the effective mass
    ###     * alp_l is the longitudinal spin-orbit coupling coefficient
    ###     * alp_t is the transverse spin-orbit coupling coefficient
    ###     * mu is the chemical potential
    ###     * V_J is an addition potential in the junction region (V = 0 in SC regions by convention)
    ###     * Gam is the Zeeman energy in the junction
    ###     * Gam_SC = Gam_SC_factor * Gam, which is the Zeeman energy in the SC regions
    ###     * Delta is the magnitude of the SC pairing in the SC regions
    ###     * phi is the phase difference between the two SC regions
    ###     * iter is the number of iteration of the algorithm to perform
    ###     * eta is the imaginary component of the energy that is used for broadening
    if nodx == 0:
        Nx = 3
    else:
        Nx = nodx+2

    Ny = int(Wj/ay_targ) # number of lattice sites in the junction region (in the y-direction)
    #ay = Wj/float(Ny+1)      # actual lattice constant
    ay = ay_targ
    Ny = Ny+2 #add one SC site on each side
    Wj_int = Ny-2

    #print("Nx", Nx)
    #print("Ny", Ny)
    #print("N", Nx*Ny)
    #print("BDG", 4*Nx*Ny)
    #print("Wj", Wj_int*ay_targ)
    #square lattice
    coor = shps.square(Nx, Ny) #square lattice
    NN = nb.NN_sqr(coor)
    NNb = nb.Bound_Arr(coor)

    Gam_SC = Gam_SC_factor * Gam

    H_J = spop.HBDG(coor, ax, ay, NN, NNb=NNb, Wj=Wj_int, cutx=nodx, cuty=nody,
    V=Vj, mu=mu, gamx=Gam, alpha=alp_l, delta=delta, phi=phi, qx=kx, meff_sc=m_eff, meff_normal=m_eff)

    Gs,Gb,sNRG_bot = bot_SC_sNRG_calc(omega=omega, Wj=Wj, nodx=nodx, ax=ax, ay_targ=ay_targ, kx=kx, m_eff=m_eff, alp_l=alp_l, alp_t=alp_t, mu=mu, Gam_SC=Gam_SC, delta=delta, phi=phi, iter=iter, eta=eta)

    Gs,Gb,sNRG_top = top_SC_sNRG_calc(omega=omega, Wj=Wj, nodx=nodx, ax=ax, ay_targ=ay_targ, kx=kx, m_eff=m_eff, alp_l=alp_l, alp_t=alp_t, mu=mu, Gam_SC=Gam_SC, delta=delta, phi=phi, iter=iter, eta=eta)

    #print(H_J.shape)
    #print(sNRG_top.shape)
    #print(sNRG_bot.shape)
    H_eff = H_J + sNRG_bot + sNRG_top
    return H_eff

def self_consistency_finder_faster(Wj, nodx, nody, ax, ay, gam, mu, Vj, alpha, delta, phi, kx, eigs_omega0, m_eff=0.023, tol=1e-3, k=4):
    delta_omega = eigs_omega0
    steps = int(delta_omega/tol) + 1
    omega = np.linspace(0, eigs_omega0, int(steps))
    omega_bands = np.zeros(omega.shape[0])

    y1 = eigs_omega0
    if y1 > 0.7*delta:
        return y1
    x1 = 0
    if eigs_omega0==0:
        return 0
    x2 = y1/50
    counter = 0
    while True:
        counter+=1
        H = Junc_eff_Ham_gen(omega=x2,Wj=Wj,nodx=nodx,nody=nody,ax=ax,ay_targ=ay,kx=kx,m_eff=m_eff,alp_l=alpha,alp_t=alpha,mu=mu,Vj=Vj,Gam=gam,Gam_SC_factor=0,delta=delta,phi=phi,iter=50,eta=0)
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        y2 = eigs[int(k/2)] - x2

        if x1==x2:
            print("x1=x2")
            print(y1, y2, tol)
            sys.exit()

        if abs(y2) < tol:
            return x2

        m = (y2-y1)/(x2-x1)
        b = y1-m*x1
        omega_c = -b/m

        y1=y2
        x1=x2
        x2 = omega_c
    return None

def gap(Wj, nodx, nody, ax, ay, gam, mu, Vj, alpha, delta, phi, muf=20, tol = 1e-3, m_eff=0.023, k=4):
    q_steps = 500
    if Vj < 0:
        VVJ = Vj
    else:
        VVJ = 0
    xi = ((const.hbar**2)*(const.e0)*(10**20)*(10**3))/(const.m0*m_eff)
    qmax = np.sqrt(2*(muf-VVJ)/xi)*1.5
    #print(qmax, np.pi/ax)
    qx = np.linspace(0, qmax, q_steps) #kx in the first Brillouin zone
    omega0_bands = np.zeros(qx.shape[0])
    for q in range(qx.shape[0]):
        print(qx.shape[0]-q)
        H = Junc_eff_Ham_gen(omega=0, Wj=Wj, nodx=nodx, nody=nody, ax=ax, ay_targ=ay, kx=qx[q], m_eff=m_eff, alp_l=alpha, alp_t=alpha, mu=mu, Vj=Vj, Gam=gam, delta=delta, phi=phi, Gam_SC_factor=0, iter=50, eta=0)

        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        omega0_bands[q] = eigs[int(k/2)]
    #plt.plot(qx, omega0_bands, c='k')
    #plt.show()

    local_min_idx = np.array(argrelextrema(omega0_bands, np.less)[0])
    local_min_idx = np.concatenate((np.array([0]), local_min_idx))
    abs_min =  omega0_bands[local_min_idx[0]]
    idx_absmin = 0
    for n in range(local_min_idx.shape[0]):
        abs_min_new = omega0_bands[local_min_idx[n]]
        if abs_min_new < abs_min:
            abs_min = abs_min_new
            idx_absmin = n
    kx_of_absmin = qx[local_min_idx[idx_absmin]]
    idx_of_absmin = local_min_idx[idx_absmin]

    print("kx at absolute minimum", kx_of_absmin)
    print("gap of omega0", omega0_bands[idx_of_absmin] )
    true_eig = self_consistency_finder_faster(Wj=Wj, nodx=nodx, nody=nody, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=kx_of_absmin, eigs_omega0=omega0_bands[idx_of_absmin], m_eff=m_eff, tol=tol, k=k)
    #true_eig2 = self_consistency_finder(gam, mu, Wj, Vj, alpha, delta, phi, kx_of_absmin, omega0_bands[idx_of_absmin], tol)
    print("gap", true_eig)
    #print("slower gap", true_eig2)
    #print(counter)
    #sys.exit()
    return true_eig, kx_of_absmin, idx_of_absmin

def solve_Ham(Ham,num,sigma,which = 'LM',Return_vecs = False):
    ### Finding "num" eigenvalues near E = sigma
    eigs,vecs = spLA.eigsh(Ham,k=num,sigma = sigma, which = which)
    idx = np.argsort(eigs)
    if Return_vecs:
        return eigs[idx], vecs[:,idx]
    else:
        return eigs[idx]

if False:
    delta=1
    omega=np.linspace(-5, 5, 1000)
    row = np.zeros(1000)
    Wj = 2000 #A
    ay_targ = 1
    alp_l = 200
    alp_t = 200
    mu = 20
    Gam_SC = 0
    phi=np.pi
    m_eff = 0.023
    ax=1
    #(omega,Wj,nodx,ay_targ,kx,m_eff,alp_l,alp_t,mu,Gam_SC,delta,phi,iter,eta)
    for i in range(omega.shape[0]):
        print(omega.shape[0]-i)
        gb_bot = top_SC_sNRG_calc(
        omega[i],Wj,0,ax,ay_targ,0,m_eff,alp_l,alp_t,mu,Gam_SC,delta,phi,50,1e-4)[1]
        row[i] = (-1/np.pi)*np.trace(gb_bot.imag)

    plt.plot(omega, row)
    plt.show()
        #row(omega) = -1/pi * np.trace(G_s/b(omega).imag))
        #eta = 1e-4
