import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as Spar
import scipy.sparse.linalg as spLA
from scipy.signal import argrelextrema

import majoranaJJ.modules.constants as const
import majoranaJJ.operators.sparse_operators as spop #sparse operators
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.operators.potentials as pot
import majoranaJJ.modules.plots as plots
import majoranaJJ.modules.finders as fndrs

"""
    Functions for generating the Hamiltonian of a uniform junction
    using finite difference method.

    The system is translation invariant in the x-direction, with the
    junction being of width W in the y-direction. A semi-infinite
    SC region is included on each side of the junction that we
    incorporate into the Green's function of the junction using
    a self-energy approach

    The self-energy is calculated using the accelerated iterative algorithm
    of Sancho (1984)
"""

def top_SC_sNRG_calc(omega, Wj, Lx, cutxT, cutxB, ax, ay, kx, m_eff, alp_l, alp_t, mu, Gam_SC, delta, phi, iter, eta):
    # Calculates the bulk and surface Greens functions of the top superconductor
    # along with the self-energy that it produces in the junction region
    # * omega is the (real) energy that goes into the Greens function
    # * W is the width of the junction
    # * ay_targ is the targeted lattice constant
    # * kx is the wavevector along the length of the junction
    # * m_eff is the effective mass
    # * alp_l is the longitudinal spin-orbit coupling coefficient
    # * alp_t is the transverse spin-orbit coupling coefficient
    # * mu is the chemical potential
    # * delta is the (complex) SC order parameter in the top SC
    # * iter is the number of iteration of the algorithm to perform
    # * eta is the imaginary component of the energy that is used for broadening

    if cutxT*cutxB == 0:
        Nx = 3
        Lx = (Nx)*ax
    else:
        Nx = int(Lx/ax)

    Wj_int = int(Wj/ay) # number of lattice sites in the junction region (in the y-direction)
    Ny = Wj_int + 2 #add one SC site on each side

    tx = -const.hbsqr_m0/(2*m_eff*ax**2)  # spin-preserving hopping strength
    ty = -const.hbsqr_m0/(2*m_eff*ay**2) # spin-preserving hopping strength
    ty_alp = alp_t/(2*ay) # spin-orbit hopping strength in the y-direction
    tx_alp = alp_l/(2*ax)  # onsite spin-orbit coupling contribution
    ep_on = -2*tx-2*ty-mu
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
            H00[i, 0] += tx*np.exp(1j*kx*Lx)
            H00[0, i] += tx*np.exp(-1j*kx*Lx)
            H00[i+Nx, 0+Nx] += tx*np.exp(1j*kx*Lx)
            H00[0+Nx, i+Nx] += tx*np.exp(-1j*kx*Lx)

            H00[i, 0+Nx] += tx_alp*np.exp(1j*kx*Lx)
            H00[0+Nx, i] += tx_alp*np.exp(-1j*kx*Lx)
            H00[i+Nx, 0 ] += -tx_alp*np.exp(1j*kx*Lx)
            H00[0, i+Nx] += -tx_alp*np.exp(-1j*kx*Lx)

            #hole
            H00[i+2*Nx, 0+2*Nx] += np.conjugate(-tx*np.exp(-1j*kx*Lx))
            H00[0+2*Nx, i+2*Nx] += np.conjugate(-tx*np.exp(1j*kx*Lx))
            H00[i+Nx+2*Nx, 0+Nx+2*Nx] += np.conjugate(-tx*np.exp(-1j*kx*Lx))
            H00[0+Nx+2*Nx, i+Nx+2*Nx] += np.conjugate(-tx*np.exp(1j*kx*Lx))

            H00[i+2*Nx, 0+Nx+2*Nx] += np.conjugate(-tx_alp*np.exp(-1j*kx*Lx))
            H00[0+Nx+2*Nx, i+2*Nx] += np.conjugate(-tx_alp*np.exp(1j*kx*Lx))
            H00[i+Nx+2*Nx, 0+2*Nx ] += np.conjugate(tx_alp*np.exp(-1j*kx*Lx))
            H00[0+2*Nx, i+Nx+2*Nx] += np.conjugate(tx_alp*np.exp(1j*kx*Lx))

    #print(H00)
    #print(H00.real)
    #print()
    #print(H00.imag)
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

def bot_SC_sNRG_calc(omega, Wj, Lx, cutxT, cutxB, ax, ay, kx, m_eff, alp_l, alp_t, mu, Gam_SC, delta, phi, iter, eta):
    # Calculates the bulk and surface Greens functions of the bottom superconductor
    # along with the self-energy that it produces in the junction region
    # * omega is the (real) energy that goes into the Greens function
    # * W is the width of the junction
    # * ay_targ is the targeted lattice constant
    # * kx is the wavevector along the length of the junction
    # * m_eff is the effective mass
    # * alp_l is the longitudinal spin-orbit coupling coefficient
    # * alp_t is the transverse spin-orbit coupling coefficient
    # * mu is the chemical potential
    # * delta is the (complex) SC order parameter in the top SC
    # * iter is the number of iteration of the algorithm to perform
    # * eta is the imaginary component of the energy that is used for broadening

    if cutxT*cutxB == 0:
        Nx = 3
        Lx = (Nx)*ax
    else:
        Nx = int(Lx/ax)

    Wj_int = int(Wj/ay) # number of lattice sites in the junction region (in the y-direction)
    Ny = Wj_int + 2 #add one SC site on each side

    tx = -const.hbsqr_m0/(2.*m_eff*ax**2)  # spin-preserving hopping strength
    ty = -const.hbsqr_m0/(2.*m_eff*ay**2) # spin-preserving hopping strength
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
            H00[i, 0] += tx*np.exp(1j*kx*Lx)
            H00[0, i] += tx*np.exp(-1j*kx*Lx)
            H00[i+Nx, 0+Nx] += tx*np.exp(1j*kx*Lx)
            H00[0+Nx, i+Nx] += tx*np.exp(-1j*kx*Lx)

            H00[i, 0+Nx] += tx_alp*np.exp(1j*kx*Lx)
            H00[0+Nx, i] += tx_alp*np.exp(-1j*kx*Lx)
            H00[i+Nx, 0 ] += -tx_alp*np.exp(1j*kx*Lx)
            H00[0, i+Nx] += -tx_alp*np.exp(-1j*kx*Lx)

            #hole
            H00[i+2*Nx, 0+2*Nx] += np.conjugate(-tx*np.exp(-1j*kx*Lx))
            H00[0+2*Nx, i+2*Nx] += np.conjugate(-tx*np.exp(1j*kx*Lx))
            H00[i+Nx+2*Nx, 0+Nx+2*Nx] += np.conjugate(-tx*np.exp(-1j*kx*Lx))
            H00[0+Nx+2*Nx, i+Nx+2*Nx] += np.conjugate(-tx*np.exp(1j*kx*Lx))

            H00[i+2*Nx, 0+Nx+2*Nx] += np.conjugate(-tx_alp*np.exp(-1j*kx*Lx))
            H00[0+Nx+2*Nx, i+2*Nx] += np.conjugate(-tx_alp*np.exp(1j*kx*Lx))
            H00[i+Nx+2*Nx, 0+2*Nx] += np.conjugate(tx_alp*np.exp(-1j*kx*Lx))
            H00[0+2*Nx, i+Nx+2*Nx] += np.conjugate(tx_alp*np.exp(1j*kx*Lx))

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

def Junc_eff_Ham_gen(omega, Wj, Lx, cutxT, cutyT, cutxB, cutyB, ax, ay, kx, m_eff, alp_l, alp_t, mu, Vj, Gam, delta, phi, Vsc=0, Gam_SC_factor=0, iter=50, eta=0, plot_junction=False):
    # Generates the effective Hamiltonian for the Junction, which includes the self-energy from both of the SC regions
    # * omega is the (real) energy that goes into the Green's function
    # * W is the width of the junction
    # * ay_targ is the targeted lattice constant
    # * kx is the wavevector along the length of the junction
    # * m_eff is the effective mass
    # * alp_l is the longitudinal spin-orbit coupling coefficient
    # * alp_t is the transverse spin-orbit coupling coefficient
    # * mu is the chemical potential
    # * Vj is an addition potential in the junction region (V = 0 in SC regions by convention)
    # * Gam is the Zeeman energy in the junction
    # * Gam_SC = Gam_SC_factor * Gam, which is the Zeeman energy in the SC regions
    # * delta is the magnitude of the SC pairing in the SC regions
    # * phi is the phase difference between the two SC regions
    # * iter is the number of iteration of the algorithm to perform
    # * eta is the imaginary component of the energy that is used for broadening
    if cutxT == 0 and cutxB == 0:
        Nx = 3
        Lx = Nx*ax
    else:
        Nx = int(Lx/ax)

    Wj_int = int(Wj/ay) # number of lattice sites in the junction region (in the y-direction)
    Ny = Wj_int + 2 #add one SC site on each side

    coor = shps.square(Nx, Ny) #square lattice
    NN = nb.NN_sqr(coor)
    NNb = nb.Bound_Arr(coor)

    Gam_SC = Gam_SC_factor * Gam

    H_J = spop.HBDG(coor=coor, ax=ax, ay=ay, NN=NN, NNb=NNb, Wj=Wj_int, cutxT=cutxT, cutyB=cutyB, cutxB=cutxB, cutyT=cutyT, Vj=Vj, Vsc=Vsc, mu=mu, gamx=Gam, alpha=alp_l, delta=delta, phi=phi, qx=kx, meff_sc=m_eff, meff_normal=m_eff, plot_junction=plot_junction)

    Gs,Gb,sNRG_bot = bot_SC_sNRG_calc(omega=omega, Wj=Wj, Lx=Lx, cutxT=cutxT, cutxB=cutxB, ax=ax, ay=ay, kx=kx, m_eff=m_eff, alp_l=alp_l, alp_t=alp_t, mu=mu, Gam_SC=Gam_SC, delta=delta, phi=phi, iter=iter, eta=eta)

    Gs,Gb,sNRG_top = top_SC_sNRG_calc(omega=omega, Wj=Wj, Lx=Lx, cutxT=cutxT, cutxB=cutxB, ax=ax, ay=ay, kx=kx, m_eff=m_eff, alp_l=alp_l, alp_t=alp_t, mu=mu, Gam_SC=Gam_SC, delta=delta, phi=phi, iter=iter, eta=eta)

    #print(H_J.shape)
    #print(sNRG_top.shape)
    #print(sNRG_bot.shape)
    H_eff = H_J + sNRG_bot + sNRG_top
    return H_eff

def self_consistency_finder(Wj, Lx, cutxT, cutyT, cutxB, cutyB, ax, ay, gam, mu, Vj, alpha, delta, phi, kx, eigs_omega0, m_eff, tol, k=4, iter=50, SOLVE = False):
    if not SOLVE and eigs_omega0 >= 5*delta:
        return eigs_omega0
    if eigs_omega0 < tol:
        return 0
    y1 = eigs_omega0
    omega1 = 0
    omega2 = y1*0.1
    n = .01
    N = 30
    while True:

        H = Junc_eff_Ham_gen(omega=omega2, Wj=Wj, Lx=Lx, cutxT=cutxT, cutyB=cutyB, cutxB=cutxB, cutyT=cutyT, ax=ax, ay=ay, kx=kx, m_eff=m_eff, alp_l=alpha, alp_t=alpha, mu=mu, Vj=Vj, Gam=gam, Gam_SC_factor=0, delta=delta, phi=phi, iter=iter, eta=0)

        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        arg = np.argmin(np.absolute(eigs))

        if omega2 == 0:
            y2 = eigs[int(k/2)]
        else:
            y2 = eigs[arg] - omega2

        #print("omega2, y2", omega2, y2)

        if omega1==omega2:
            print("omega1==omega2")
            print(y1, y2, tol)
            return eigs_omega0
        if abs(y2) < tol:
            #print('here')
            if abs(omega2) > eigs_omega0:
                return eigs_omega0
            else:
                return abs(omega2)
        m = (y2-y1)/(omega2-omega1)
        b = y1-m*omega1
        omega_c = -b/m
        y1=y2
        omega1 = omega2
        omega2 = omega_c

        if abs(omega2) > eigs_omega0 or y2 < -0.2*delta or abs(omega2) >= delta:
            #print('here')
            omega2 = abs(eigs_omega0*(1-np.exp(-n/N)))
            n += 0.5
            if abs(omega2-eigs_omega0) < 1e-2 or (abs(omega2-delta) < 1e-2 and not SOLVE):
                return eigs_omega0
    return None

def gap(
    Wj, Lx, cutxT, cutyT, cutxB, cutyB, ax, ay,
    gam, mu, Vj, alpha, delta, phi,
    m_eff=0.026, k=4, muf=10, targ_steps=2000, tol=1e-7, n_avg=5, iter=50,
    PLOT=False
    ):
    n1, n2 = fndrs.step_finder(targ_steps, n_avg=n_avg)
    n1 = 500#1500
    n2 = 80
    print("steps", n1, n2)
    VVJ = 0
    if Vj < 0:
        VVJ = Vj
    if muf < 1:
        muf = 5
    qmax = np.sqrt(2*(muf-VVJ)*m_eff/const.hbsqr_m0)*1.25
    if qmax >= np.pi/Lx or cutxT != 0 or cutxB != 0:
        qmax = np.pi/(Lx)
    qmax = np.pi/(Lx)
    qx = np.linspace(0, qmax, n1) #kx in the first Brillouin zone
    #print(qmax,  np.pi/(Lx))
    omega0_bands = np.zeros(n1)
    for q in range(n1):
        if (n1-q)%100 == 0:
            print(n1-q)
        H = Junc_eff_Ham_gen(omega=0, Wj=Wj, Lx=Lx, cutxT=cutxT, cutyB=cutyB, cutxB=cutxB, cutyT=cutyT, ax=ax, ay=ay, kx=qx[q], m_eff=m_eff, alp_l=alpha, alp_t=alpha, mu=mu, Vj=Vj, Gam=gam, delta=delta, phi=phi, Gam_SC_factor=0, iter=iter, eta=0)

        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        omega0_bands[q] = eigs[int(k/2)]

    if PLOT:
        plt.plot(qx, omega0_bands, c='k')
        plt.show()

    local_min_idx = np.array(argrelextrema(omega0_bands, np.less)[0])
    print(local_min_idx.size, "Local minima found at kx = {} w/ energies {}".format(qx[local_min_idx], omega0_bands[local_min_idx]))

    mins = []
    kx_of_mins = []

    #checking edge cases
    print("Energy of k=0 w=0: ", omega0_bands[0])
    print("Energy of k=pi/lx w=0: ", omega0_bands[-1])
    #print(omega0_bands[0]/5, min(omega0_bands[local_min_idx]))

    if local_min_idx.size == 0:
        if omega0_bands[0]/5 <= omega0_bands[-1]:
            true_gap_of_k0 = self_consistency_finder(Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=0, eigs_omega0=omega0_bands[0], m_eff=m_eff, tol=tol, k=k, iter=iter, SOLVE=False)
            mins.append(true_gap_of_k0)
            kx_of_mins.append(qx[0])
        else:
            true_gap_of_k0 = omega0_bands[0]
            mins.append(omega0_bands[0])
            kx_of_mins.append(qx[0])
        if omega0_bands[-1]/5 <= omega0_bands[0]:
            true_gap_of_kedge = self_consistency_finder(Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=qx[-1], eigs_omega0=omega0_bands[-1], m_eff=m_eff, tol=tol, k=k, iter=iter, SOLVE=False)
            mins.append(true_gap_of_kedge)
            kx_of_mins.append(qx[-1])
        else:
            true_gap_of_kedge = omega0_bands[-1]
            mins.append(omega0_bands[-1])
            kx_of_mins.append(qx[-1])
    else:
        if omega0_bands[0]/5 <= min(omega0_bands[local_min_idx]):
            true_gap_of_k0 = self_consistency_finder(Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=0, eigs_omega0=omega0_bands[0], m_eff=m_eff, tol=tol, k=k, iter=iter)
            mins.append(true_gap_of_k0)
            kx_of_mins.append(qx[0])
        else:
            true_gap_of_k0 = omega0_bands[0]
            mins.append(omega0_bands[0])
            kx_of_mins.append(qx[0])
        if omega0_bands[-1]/5 <= min(omega0_bands[local_min_idx]):
            true_gap_of_kedge = self_consistency_finder(Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=qx[-1], eigs_omega0=omega0_bands[-1], m_eff=m_eff, tol=tol, k=k, iter=iter)
            mins.append(true_gap_of_kedge)
            kx_of_mins.append(qx[-1])
        else:
            true_gap_of_kedge = omega0_bands[-1]
            mins.append(omega0_bands[-1])
            kx_of_mins.append(qx[-1])

    print("Energy at kx=0: ", true_gap_of_k0)
    print("Energy at k_max:  ", true_gap_of_kedge)

    slfCON = True
    for i in range(local_min_idx.shape[0]):
        if (omega0_bands[local_min_idx[i]] > 5*omega0_bands[0]) or omega0_bands[local_min_idx[i]] > 5*omega0_bands[-1] or omega0_bands[local_min_idx[i]] > 5*min(omega0_bands[local_min_idx]):
            pass
        else:
            kx_c = qx[local_min_idx[i]]
            if slfCON:
                if local_min_idx[i]-2 < 0:
                    kx_lower = qx[0]
                else:
                    kx_lower = qx[local_min_idx[i]-2]
                if local_min_idx[i]+2 >= qx.shape[0]:
                    kx_higher = qx[-1]
                else:
                    kx_higher = qx[local_min_idx[i]+2]
            else:
                kx_lower = qx[local_min_idx[i]-1]
                kx_higher = qx[local_min_idx[i]+1]

            deltaq = kx_higher - kx_lower
            kx_finer = np.linspace(kx_lower, kx_higher, n2)
            omega0_arr_finer = np.zeros((kx_finer.size))
            for j in range(kx_finer.shape[0]):
                if (kx_finer.shape[0] - j)%10 == 0:
                    print(kx_finer.shape[0] - j)
                H = Junc_eff_Ham_gen(omega=0, Wj=Wj,Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, kx=kx_finer[j], m_eff=m_eff, alp_l=alpha, alp_t=alpha, mu=mu, Vj=Vj, Gam=gam, delta=delta, phi=phi, Gam_SC_factor=0, iter=iter, eta=0)

                eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
                idx_sort = np.argsort(eigs)
                eigs = eigs[idx_sort]
                if slfCON:
                    EIG = self_consistency_finder(Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB, cutyB=cutyB, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=kx_finer[j], eigs_omega0=eigs[int(k/2)], m_eff=m_eff, tol=1e-7, k=k, iter=iter)
                else:
                    EIG = eigs[int(k/2)]
                omega0_arr_finer[j] = EIG

            if PLOT:
                plt.plot(kx_finer, omega0_arr_finer, c='k')
                plt.show()

            GAP, IDX = fndrs.minima(omega0_arr_finer)
            true_eig = GAP
            #true_eig = self_consistency_finder(Wj=Wj, Lx=Lx, cutxT=cutxT, cutyT=cutyT, cutxB=cutxB,  cutyB=cutyB, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=kx_finer[IDX], eigs_omega0=GAP, m_eff=m_eff, tol=tol, k=k, iter=iter)
            print("Minimum {} energy: {}".format(i+1, true_eig))
            print("Kx of minium energy {}: {}: ".format(i+1, kx_finer[IDX]))
            mins.append(true_eig)
            kx_of_mins.append(kx_finer[IDX])

    mins = np.array(mins)
    true_gap, idx = fndrs.minima(mins)
    kx_of_gap = kx_of_mins[idx]

    if local_min_idx.shape[0] != 0 and true_gap > min(omega0_bands[local_min_idx]):
        print("Gap: ", min(omega0_bands[local_min_idx]))
        print("kx of gap: ", kx_of_gap)
        print()
        return min(omega0_bands[local_min_idx]), kx_of_gap
    else:
        print("Gap: ", true_gap)
        print("kx of gap: ", kx_of_gap)
        print()
        return true_gap, kx_of_gap

if False:
    #plotting local density of states
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

if False:#False True
    #plotting bands
    m_eff=0.026
    tol=1e-8

    ax = 50 #lattice spacing in x-direction: [A]
    ay = 50 #lattice spacing in y-direction: [A]
    Nx = 12 #Number of lattice sites along x-direction
    Wj = 1000 #Junction region [A]
    cutx = 4 #width of nodule
    cuty = 8 #height of nodule
    Lx = Nx*ax

    alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
    phi = 0*np.pi #SC phase difference
    delta = 0.3 #Superconducting Gap: [meV]
    Vsc = 0 #SC potential: [meV]
    Vj = -40 #Junction potential: [meV]
    mu = 11.12
    gam = 2.404 #mev

    steps = 300
    if Vj < 0:
        VVJ = Vj
    else:
        VVJ = 0

    if mu < 1:
        muf = 5
    else:
        muf= mu
    qmax = np.sqrt(2*(muf-VVJ)*m_eff/const.hbsqr_m0)*1.25
    print(qmax)
    kx = np.linspace(0, np.pi/Lx, steps)
    k = 4
    #kx = np.linspace(0.004, 0.0042, steps)
    omega0_bands = np.zeros((k, kx.shape[0]))
    true_bands = np.zeros(kx.shape[0])

    for i in range(kx.shape[0]):
        print(2*omega0_bands.shape[1]-i, kx[i])

        H = Junc_eff_Ham_gen(omega=0, Wj=Wj, Lx=Lx, cutxT=cutx, cutyT=cuty, cutxB=cutx, cutyB=cuty, ax=ax, ay=ay, kx=kx[i], m_eff=m_eff, alp_l=alpha, alp_t=alpha, mu=mu, Vj=Vj, Gam=gam, delta=delta, phi=phi, Vsc=0, Gam_SC_factor=0, iter=50, eta=0, plot_junction=False)
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        #print(eigs)
        #arg = np.argmin(np.absolute(eigs))
        #print(arg)
        omega0_bands[:, i] = eigs[:]

    for i in range(k):
        plt.plot(kx, omega0_bands[i, :], c='b')
    plt.title('Omega0 bands')
    plt.show()
    for i in range(kx.shape[0]):
        print(omega0_bands.shape[1]-i)
        true_eig = self_consistency_finder(Wj=Wj, Lx=Lx, cutxT=cutx, cutyT=cuty, cutxB=cutx, cutyB=cuty, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=kx[i], eigs_omega0=omega0_bands[int(k/2), i], m_eff=m_eff, tol=tol, k=k)
        true_bands[i] = true_eig

    plt.plot(kx, true_bands)
    plt.plot(kx, 0*kx, c='k', ls='--')
    plt.title('true band calculated from Greens function and self consistency finder', loc = 'center', wrap = True)
    plt.show()
    sys.exit()

if False: #True False
    #plotting E-w vs w
    m_eff=0.026
    k=4

    ax = 50 #lattice spacing in x-direction: [A]
    ay = 50 #lattice spacing in y-direction: [A]
    Nx = 12 #Number of lattice sites along x-direction
    Wj = 1000 #Junction region [A]
    nodx = 4 #width of nodule
    nody = 8 #height of nodule
    Lx = Nx*ax

    alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
    phi = np.pi #SC phase difference
    delta = 0.3 #Superconducting Gap: [meV]
    Vsc = 0 #SC potential: [meV]
    Vj = -40 #Junction potential: [meV]
    mu = 2.69
    gam = 1 #mev

    if nodx == 0:
        Nx = 3
        Lx = (Nx)*ax
    else:
        Nx = int(Lx/ax)

    Wj_int = int(Wj/ay) # number of lattice sites in the junction region (in the y-direction)
    Ny = 800
    coor = shps.square(Nx, Ny) #square lattice
    NN = nb.NN_sqr(coor)
    NNb = nb.Bound_Arr(coor)
    H_sp = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj=Wj_int, cutx=nodx, cuty=nody, Vj=Vj, mu=mu, gamx=gam, alpha=alpha, delta=delta, phi=phi, meff_normal=m_eff, meff_sc=m_eff, qx=np.pi/Lx)
    #eigs, vecs = spLA.eigsh(H_sp, k=k, sigma=0, which='LM')
    #idx_sort = np.argsort(eigs)
    #eigs = eigs[idx_sort]
    #print("Real eig =", eigs[int(k/2)])

    w_steps = 200
    w = np.linspace(-2*delta*0, 5*delta, w_steps) #kx in the first Brillouin zone
    E = np.zeros(w_steps)
    for i in range(w_steps):
        print(w_steps-i)
        H = Junc_eff_Ham_gen(omega=w[i], Wj=Wj, Lx=Lx, nodx=nodx, nody=nody, ax=ax, ay=ay, kx=np.pi/Lx, m_eff=m_eff, alp_l=alpha, alp_t=alpha, mu=mu, Vj=Vj, Gam=gam, delta=delta, phi=phi, Gam_SC_factor=0, iter=50, eta=0)

        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        arg = np.argmin(np.absolute(eigs))
        if w[i] == 0:
            E[i] = eigs[int(k/2)]
        else:
            E[i] = eigs[arg]

    plt.plot(w, E-w, c='b')
    plt.plot(w, w*0, c='k', ls='--')
    plt.show()
    sys.exit()

if False:#False True
    #plotting E vs gam
    m_eff=0.026
    tol=1e-8
    k=4

    ax = 50 #lattice spacing in x-direction: [A]
    ay = 50 #lattice spacing in y-direction: [A]
    Nx = 10 #Number of lattice sites along x-direction
    Wj = 300 #Junction region [A]
    nodx = 4 #width of nodule
    nody = 2 #height of nodule
    Lx = Nx*ax

    alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
    phi = np.pi #SC phase difference
    delta = 1 #Superconducting Gap: [meV]
    Vsc = 0 #SC potential: [meV]
    Vj = -20 #Junction potential: [meV]
    mu = 15.3
    steps = 100
    gam = np.linspace(0, 3, steps) #mev

    kx = 0
    k = 20
    #kx = np.linspace(0.004, 0.0042, steps)
    omega0_bands = np.zeros((k, gam.shape[0]))
    true_bands = np.zeros((k, gam.shape[0]))

    for i in range(gam.shape[0]):
        print(2*gam.shape[0]-i, gam[i])

        H = Junc_eff_Ham_gen(omega=0, Wj=Wj, Lx=Lx, nodx=nodx, nody=nody, ax=ax, ay=ay, kx=kx, m_eff=m_eff, alp_l=alpha, alp_t=alpha, mu=mu, Vj=Vj, Gam=gam[i], Gam_SC_factor=0, delta=delta, phi=phi, iter=50, eta=0)

        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        #print(eigs)
        #arg = np.argmin(np.absolute(eigs))
        #print(arg)
        omega0_bands[:, i] = eigs[:]

    for i in range(k):
        plt.plot(gam, omega0_bands[i, :], c='b')
    plt.title('Omega0 bands')
    plt.show()
    for i in range(omega0_bands.shape[0]):
        print(omega0_bands.shape[0]-i)
        true_eig = self_consistency_finder(Wj=Wj, Lx=Lx,nodx=nodx, nody=nody, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=kx[i], eigs_omega0=omega0_bands[i], m_eff=m_eff, tol=tol, k=k)
        true_bands[:, i] = true_eig

    plt.plot(kx, true_bands)
    plt.plot(kx, 0*kx, c='k', ls='--')
    plt.title('true band calculated from Greens function and self consistency finder', loc = 'center', wrap = True)
    plt.show()
    sys.exit()

if False:#False True
    #plotting SOC bands
    m_eff=0.026
    tol=1e-8
    k=4

    ax = 50 #lattice spacing in x-direction: [A]
    ay = 50 #lattice spacing in y-direction: [A]
    Nx = 15 #Number of lattice sites along x-direction
    Wj = 1000 #Junction region [A]
    nodx = 5 #width of nodule
    nody = 8 #height of nodule
    Lx = Nx*ax

    alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
    phi = 0*np.pi #SC phase difference
    delta = 0.3 #Superconducting Gap: [meV]
    Vsc = 0 #SC potential: [meV]
    Vj = -20 #Junction potential: [meV]
    mu = 6.08
    gam = 1 #mev

    steps = 100
    kx = np.linspace(0, np.pi/Lx, steps)
    k = 20
    #kx = np.linspace(0.004, 0.0042, steps)
    omega0_bands = np.zeros((k, kx.shape[0]))
    true_bands = np.zeros((k, kx.shape[0]))

    for i in range(kx.shape[0]):
        print(2*omega0_bands.shape[1]-i, kx[i])

        H = Junc_eff_Ham_gen(omega=0, Wj=Wj, Lx=Lx, nodx=nodx, nody=nody, ax=ax, ay=ay, kx=kx[i], m_eff=m_eff, alp_l=alpha, alp_t=alpha, mu=mu, Vj=Vj, Gam=gam, Gam_SC_factor=0, delta=delta, phi=phi, iter=50, eta=0)
        S = int(H.shape[0]/2)
        H = (H[:S, :])[:, :S]
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        #print(eigs)
        #arg = np.argmin(np.absolute(eigs))
        #print(arg)
        omega0_bands[:, i] = eigs[:]

    for i in range(k):
        plt.plot(kx, omega0_bands[i, :], c='b')
    plt.title('Omega0 bands')
    plt.show()
    for i in range(omega0_bands.shape[0]):
        print(omega0_bands.shape[0]-i)
        true_eig = self_consistency_finder(Wj=Wj, Lx=Lx,nodx=nodx, nody=nody, ax=ax, ay=ay, gam=gam, mu=mu, Vj=Vj, alpha=alpha, delta=delta, phi=phi, kx=kx[i], eigs_omega0=omega0_bands[i], m_eff=m_eff, tol=tol, k=k)
        true_bands[:, i] = true_eig

    plt.plot(kx, true_bands)
    plt.plot(kx, 0*kx, c='k', ls='--')
    plt.title('true band calculated from Greens function and self consistency finder', loc = 'center', wrap = True)
    plt.show()
    sys.exit()
