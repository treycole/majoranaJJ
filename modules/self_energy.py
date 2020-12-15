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

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as Spar
import scipy.sparse.linalg as spLA
from scipy.signal import argrelextrema
from majoranaJJ.modules import parameters as par
import majoranaJJ.modules.constants as params
np.set_printoptions(linewidth = 500)

def Junc_Ham_gen(W,ay_targ,kx,m_eff,alp_l,alp_t,mu,V_J,Gam):
    ### Generates the BdG Hamiltonian of the isolated junction region (i.e. no SC included).
    ###     * W is the width of the junction
    ###     * ay_targ is the targeted lattice constant
    ###     * kx is the wavevector along the length of the junction
    ###     * m_eff is the effective mass
    ###     * alp_l is the longitudinal spin-orbit coupling coefficient
    ###     * alp_t is the transverse spin-orbit coupling coefficient
    ###     * mu is the chemical potential
    ###     * V_J is an addition potential in the junction region (V = 0 in SC regions by convention)
    ###     * Gam is the Zeeman energy

    N = int(W/ay_targ) - 1 # number of lattice sites in the junction (in the y-direction)
    ay = W/float(N+1)      # actual lattice constant

    t = -1000.*par.hbm0/(2.*m_eff*ay**2)        # spin-preserving hopping strength
    t_alp = alp_t / (2.*ay)                     # spin-orbit hopping strength in the y-direction
    alp_onsite = kx * alp_l                     # onsite spin-orbit coupling contribution
    ep_kx = 1000.*par.hbm0 * kx**2/(2.*m_eff)   # kinetic energy from momentum in x-direction

    row = []; col = []; data = []
    for i in range(N):

        ### onsite terms
        row.append(i + 0); col.append(i + 0); data.append(-2*t - mu + ep_kx + V_J)
        row.append(i + N); col.append(i + N); data.append(-2*t - mu + ep_kx + V_J)
        row.append(i + 0); col.append(i + N); data.append(-1j*alp_onsite + Gam)
        row.append(i + N); col.append(i + 0); data.append(1j*alp_onsite + Gam)

        row.append(i + 2*N); col.append(i + 2*N); data.append(-(-2*t - mu + ep_kx + V_J))
        row.append(i + 3*N); col.append(i + 3*N); data.append(-(-2*t - mu + ep_kx + V_J))
        row.append(i + 2*N); col.append(i + 3*N); data.append(1j*alp_onsite - Gam)
        row.append(i + 3*N); col.append(i + 2*N); data.append(-1j*alp_onsite - Gam)

        ### nearest neighbor terms
        if i != N-1:
            row.append(i+1 + 0); col.append(i + 0); data.append(t)
            col.append(i+1 + 0); row.append(i + 0); data.append(t)

            row.append(i+1 + N); col.append(i + N); data.append(t)
            col.append(i+1 + N); row.append(i + N); data.append(t)

            row.append(i+1 + 2*N); col.append(i + 2*N); data.append(-t)
            col.append(i+1 + 2*N); row.append(i + 2*N); data.append(-t)

            row.append(i+1 + 3*N); col.append(i + 3*N); data.append(-t)
            col.append(i+1 + 3*N); row.append(i + 3*N); data.append(-t)

            row.append(i+1 + 0); col.append(i + N); data.append(1j*t_alp)
            col.append(i+1 + 0); row.append(i + N); data.append(-1j*t_alp)

            row.append(i+1 + N); col.append(i + 0); data.append(1j*t_alp)
            col.append(i+1 + N); row.append(i + 0); data.append(-1j*t_alp)

            row.append(i+1 + 2*N); col.append(i + 3*N); data.append(1j*t_alp)
            col.append(i+1 + 2*N); row.append(i + 3*N); data.append(-1j*t_alp)

            row.append(i+1 + 3*N); col.append(i + 2*N); data.append(1j*t_alp)
            col.append(i+1 + 3*N); row.append(i + 2*N); data.append(-1j*t_alp)

    H_J = Spar.csc_matrix((data,(row,col)),shape = (4*N,4*N),dtype = 'complex')
    return H_J

def top_SC_sNRG_calc(omega,W,ay_targ,kx,m_eff,alp_l,alp_t,mu,Gam_SC,Delta,iter,eta):
    ### Calculates the bulk and surface Greens functions of the top superconductor
    ### along with the self-energy that it produces in the junction region
    ###     * omega is the (real) energy that goes into the Greens function
    ###     * W is the width of the junction
    ###     * ay_targ is the targeted lattice constant
    ###     * kx is the wavevector along the length of the junction
    ###     * m_eff is the effective mass
    ###     * alp_l is the longitudinal spin-orbit coupling coefficient
    ###     * alp_t is the transverse spin-orbit coupling coefficient
    ###     * mu is the chemical potential
    ###     * Delta is the (complex) SC order parameter in the top SC
    ###     * iter is the number of iteration of the algorithm to perform
    ###     * eta is the imaginary component of the energy that is used for broadening

    N = int(W/ay_targ) - 1 # number of lattice sites in the junction region (in the y-direction)
    ay = W/float(N+1)      # actual lattice constant

    t = -1000.*par.hbm0/(2.*m_eff*ay**2)        # spin-preserving hopping strength
    t_alp = alp_t / (2.*ay)                     # spin-orbit hopping strength in the y-direction
    alp_on = kx * alp_l                         # onsite spin-orbit coupling contribution
    ep_kx = 1000.*par.hbm0 * kx**2/(2.*m_eff)   # kinetic energy from momentum in x-direction
    ep_on = -2*t - mu + ep_kx
    Dc = np.conjugate(Delta)

    ### Onsite Hamiltonian matrix
    H00 = np.array([
                    [ep_on,-1j*alp_on + Gam_SC,0.,Delta],
                    [1j*alp_on + Gam_SC,ep_on,-Delta,0.],
                    [0.,-Dc,-ep_on,1j*alp_on - Gam_SC],
                    [Dc,0.,-1j*alp_on - Gam_SC,-ep_on]
    ],dtype = 'complex')

    ### Hopping Hamiltonian matrix
    H10 = np.array([
                    [t,1j*t_alp,0.,0.],
                    [1j*t_alp,t,0.,0.],
                    [0,0.,-t,1j*t_alp],
                    [0,0.,1j*t_alp,-t],
    ],dtype = 'complex')
    H01 = np.conjugate(np.transpose(H10))

    ### Identity matrix
    I = np.array([
                   [1.,0.,0.,0.],
                   [0.,1.,0.,0.],
                   [0.,0.,1.,0.],
                   [0.,0.,0.,1.]
                  ],dtype = 'complex')

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

    sNRG_4b4 = np.dot(H01,np.dot(G_s,H10))

    row = []; col = []; data = []
    for m in range(4):
        for n in range(4):
            row.append(N-1 + m*N); col.append(N-1 + n*N); data.append(sNRG_4b4[m,n])
    sNRG_mtx = Spar.csc_matrix((data,(row,col)),shape = (4*N,4*N),dtype = 'complex')

    return G_s ,G_b, sNRG_mtx

def bot_SC_sNRG_calc(omega,W,ay_targ,kx,m_eff,alp_l,alp_t,mu,Gam_SC,Delta,iter,eta):
    ### Calculates the bulk and surface Greens functions of the bottom superconductor
    ### along with the self-energy that it produces in the junction region
    ###     * omega is the (real) energy that goes into the Greens function
    ###     * W is the width of the junction
    ###     * ay_targ is the targeted lattice constant
    ###     * kx is the wavevector along the length of the junction
    ###     * m_eff is the effective mass
    ###     * alp_l is the longitudinal spin-orbit coupling coefficient
    ###     * alp_t is the transverse spin-orbit coupling coefficient
    ###     * mu is the chemical potential
    ###     * Delta is the (complex) SC order parameter in the top SC
    ###     * iter is the number of iteration of the algorithm to perform
    ###     * eta is the imaginary component of the energy that is used for broadening

    N = int(W/ay_targ) - 1 # number of lattice sites in the junction region (in the y-direction)
    ay = W/float(N+1)      # actual lattice constant

    t = -1000.*par.hbm0/(2.*m_eff*ay**2)        # spin-preserving hopping strength
    t_alp = alp_t / (2.*ay)                     # spin-orbit hopping strength in the y-direction
    alp_on = kx * alp_l                         # onsite spin-orbit coupling contribution
    ep_kx = 1000.*par.hbm0 * kx**2/(2.*m_eff)   # kinetic energy from momentum in x-direction
    ep_on = -2*t - mu + ep_kx
    Dc = np.conjugate(Delta)

    ### Onsite Hamiltonian matrix
    H00 = np.array([
                    [ep_on,-1j*alp_on + Gam_SC,0.,Delta],
                    [1j*alp_on + Gam_SC,ep_on,-Delta,0.],
                    [0.,-Dc,-ep_on,1j*alp_on - Gam_SC],
                    [Dc,0.,-1j*alp_on - Gam_SC,-ep_on]
    ],dtype = 'complex')

    ### Hopping Hamiltonian matrix
    H01 = np.array([
                    [t,1j*t_alp,0.,0.],
                    [1j*t_alp,t,0.,0.],
                    [0,0.,-t,1j*t_alp],
                    [0,0.,1j*t_alp,-t],
    ],dtype = 'complex')
    H10 = np.conjugate(np.transpose(H01))

    ### Identity matrix
    I = np.array([
                   [1.,0.,0.,0.],
                   [0.,1.,0.,0.],
                   [0.,0.,1.,0.],
                   [0.,0.,0.,1.]
                  ],dtype = 'complex')

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

    sNRG_4b4 = np.dot(H01,np.dot(G_s,H10))

    row = []; col = []; data = []
    for m in range(4):
        for n in range(4):
            row.append(0 + m*N); col.append(0 + n*N); data.append(sNRG_4b4[m,n])
    sNRG_mtx = Spar.csc_matrix((data,(row,col)),shape = (4*N,4*N),dtype = 'complex')

    return G_s ,G_b, sNRG_mtx

def Junc_eff_Ham_gen(omega,W,ay_targ,kx,m_eff,alp_l,alp_t,mu,V_J,Gam,Gam_SC_factor,Delta,phi,iter,eta):
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

    Gam_SC = Gam_SC_factor * Gam

    H_J = Junc_Ham_gen(W,ay_targ,kx,m_eff,alp_l,alp_t,mu,V_J,Gam) # isolated junction Hamiltonian
    Gs,Gb,sNRG_bot = bot_SC_sNRG_calc(omega,W,ay_targ,kx,m_eff,alp_l,alp_t,mu,Gam_SC,Delta,iter,eta)
    Gs,Gb,sNRG_top = top_SC_sNRG_calc(omega,W,ay_targ,kx,m_eff,alp_l,alp_t,mu,Gam_SC,Delta*np.exp(1j*phi),iter,eta)
    H_eff = H_J + sNRG_bot + sNRG_top
    return H_eff

def solve_Ham(Ham,num,sigma,which = 'LM',Return_vecs = False):
    ### Finding "num" eigenvalues near E = sigma
    eigs,vecs = spLA.eigsh(Ham,k=num,sigma = sigma, which = which)
    idx = np.argsort(eigs)
    if Return_vecs:
        return eigs[idx], vecs[:,idx]
    else:
        return eigs[idx]
#for each point in parameter space we now have the kx value of the absolute minimum of the band structure
#now, around this kx value we know that the true minimum is close
#omega needs to be scanned from 0 to the eigenvalue at that k value and omega=0
def gap(ay, gam, mu, Vj, Wj, alpha, delta, phi, muf=20, tol = 1e-3, m_eff=0.023, k=4):
    q_steps = 500
    if Vj < 0:
        VVJ = Vj
    else:
        VVJ = 0
    qmax = np.sqrt(2*(muf-VVJ)/params.xi)*1.5
    #print(qmax, np.pi/ax)
    qx = np.linspace(0, qmax, q_steps) #kx in the first Brillouin zone
    omega0_bands = np.zeros(qx.shape[0])
    for q in range(qx.shape[0]):
        print(qx.shape[0]-q)
        H = Junc_eff_Ham_gen(omega=0,W=Wj,ay_targ=ay,kx=qx[q],m_eff=m_eff,alp_l=alpha,alp_t=alpha,mu=mu,V_J=Vj,Gam=gam,Gam_SC_factor=0,Delta=delta,phi=phi,iter=50,eta=0)

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
    #print("kx at absolute minimum", kx_of_absmin)
    #print("gap of omega0", omega0_bands[idx_of_absmin] )
    true_eig = self_consistency_finder_faster(ay, gam, mu, Wj, Vj, alpha, delta, phi, kx_of_absmin, omega0_bands[idx_of_absmin], tol)
    #true_eig2 = self_consistency_finder(gam, mu, Wj, Vj, alpha, delta, phi, kx_of_absmin, omega0_bands[idx_of_absmin], tol)
    print("gap", true_eig)
    #print("slower gap", true_eig2)
    #print(counter)
    #sys.exit()
    return true_eig, kx_of_absmin, idx_of_absmin

def self_consistency_finder(ay, gam, mu, Wj, Vj, alpha, delta, phi, kx, eigs_omega0, tol=1e-3, k=4):
    true_eig = None
    delta_omega = eigs_omega0
    steps = int(delta_omega/tol) + 1
    omega = np.linspace(0, eigs_omega0, int(steps))
    omega_bands = np.zeros(omega.shape[0])
    for w in range(omega.shape[0]):
        #print(omega.shape[0]-w)
        H = Junc_eff_Ham_gen(omega=omega[w],W=Wj,ay_targ=ay,kx=kx,m_eff=0.023,alp_l=alpha,alp_t=alpha,mu=mu,V_J=Vj,Gam=gam,Gam_SC_factor=0,Delta=delta,phi=phi,iter=50,eta=0)
        eigs, vecs = spLA.eigsh(H, k=k, sigma=0, which='LM')
        idx_sort = np.argsort(eigs)
        eigs = eigs[idx_sort]
        omega_bands[w] = eigs[int(k/2)]
        #print(omega[w], abs(eigs[int(k/2)] - omega[w]))
        if abs(eigs[int(k/2)] - omega[w]) < tol:
            true_eig = eigs[int(k/2)]
            break
    #plt.plot(omega, omega_bands-omega, c='k')
    #plt.plot(omega, omega, c='b')
    #plt.show()
    return true_eig

def self_consistency_finder_faster(ay, gam, mu, Wj, Vj, alpha, delta, phi, kx, eigs_omega0, tol=1e-3, k=4):
    delta_omega = eigs_omega0
    steps = int(eigs_omega0/tol) + 1
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
        H = Junc_eff_Ham_gen(omega=x2,W=Wj,ay_targ=ay,kx=kx,m_eff=0.023,alp_l=alpha,alp_t=alpha,mu=mu,V_J=Vj,Gam=gam,Gam_SC_factor=0,Delta=delta,phi=phi,iter=50,eta=0)
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

if False:
    ### Testing the isolated junction BdG spectrum

    W = 50. * 10.
    ay_targ = 5. * 10.
    kx = 0.00
    m_eff = 0.023
    alp_l = 100.
    alp_t = 100.
    mu = 0.
    V_J = 1.
    Gam = .0

    Ham = Junc_Ham_gen(W,ay_targ,kx,m_eff,alp_l,alp_t,mu,V_J,Gam)
    #print np.around(Ham.todense().real,decimals = 2)
    #print "\n"
    #print np.around(Ham.todense().imag,decimals = 2)

    num = 20
    kx = np.linspace(-0.01,0.01,1000)
    eig_arr = np.zeros((kx.size,num))
    for i in range(kx.size):
        if i % 20 == 0:
            print(kx.size - i)
        Ham = Junc_Ham_gen(W,ay_targ,kx[i],m_eff,alp_l,alp_t,mu,V_J,Gam)
        eig_arr[i,:] = solve_Ham(Ham,num,0.)

    for i in range(eig_arr.shape[1]):
        plt.plot(kx,eig_arr[:,i], c= 'k')
    plt.grid()
    plt.show()

if False:
    ### Testing the density of states of the bulk superconductor

    W = 8. * 10.
    ay_targ = 2. * 10.
    kx = 0.00
    m_eff = 0.023
    alp_l = 500.
    alp_t = 500.
    mu = 5.
    V_J = 1.
    Gam = .0
    Delta = .5
    iter = 50
    eta = 1.e-10

    omega = np.linspace(.0,10.,10000)
    DOS_b = np.zeros(omega.size)
    DOS_s = np.zeros(omega.size)
    #sNRG_arr = np.zeros((omega.size,4,4),dtype = 'complex')
    #G_s_arr = np.zeros((omega.size,4,4),dtype = 'complex')

    for i in range(omega.size):
        if i % 100 == 0:
            print(omega.size - i)
        G_s,G_b,sNRG = top_SC_sNRG_calc(omega[i],W,ay_targ,kx,m_eff,alp_l,alp_t,mu,Gam,Delta,iter,eta)
        print(np.around(sNRG.todense().real,decimals = 5))
        print('\n')
        print(np.around(sNRG.todense().imag,decimals = 5))
        sys.exit()
        DOS_b[i] = (-1./np.pi)*np.trace(G_b.imag)
        DOS_s[i] = (-1./np.pi)*np.trace(G_s.imag)
        #sNRG_arr[i,:,:] = sNRG[:,:]
        #G_s_arr[i,:,:] = G_s[:,:]

    plt.plot(omega,DOS_b,c = 'k')
    plt.plot(omega,DOS_s,c = 'r')
    plt.show()

if False:
    ### Testing the BdG spectrum of H_eff(omega = 0), H_eff(omega = Delta / 3.), H_eff(omega = -Delta / 3.)

    W = 100. * 10.
    ay_targ = 5. * 10.
    kx = 0.00
    m_eff = 0.023
    alp_l = 100.
    alp_t = 100.
    mu = 10.
    V_J = 1.
    Gam = 1.
    Gam_SC_factor = 0.0
    Delta = .5
    phi = 0.
    iter = 50
    eta = 0.
    omega_0 = 0.
    omega_1 = Delta/3.

    ### Finding spectra
    num = 20
    kx = np.linspace(0.,0.01,500)
    eig_arr_0 = np.zeros((kx.size,num))
    eig_arr_1 = np.zeros((kx.size,num))
    eig_arr_1m = np.zeros((kx.size,num))
    for i in range(kx.size):
        if i % 20 == 0:
            print(kx.size - i)
        Ham_eff = Junc_eff_Ham_gen(omega_0,W,ay_targ,kx[i],m_eff,alp_l,alp_t,mu,V_J,Gam,Gam_SC_factor,Delta,phi,iter,eta)
        eig_arr_0[i,:] = solve_Ham(Ham_eff,num,0.)
        Ham_eff = Junc_eff_Ham_gen(omega_1,W,ay_targ,kx[i],m_eff,alp_l,alp_t,mu,V_J,Gam,Gam_SC_factor,Delta,phi,iter,eta)
        eig_arr_1[i,:] = solve_Ham(Ham_eff,num,0.)
        Ham_eff = Junc_eff_Ham_gen(-omega_1,W,ay_targ,kx[i],m_eff,alp_l,alp_t,mu,V_J,Gam,Gam_SC_factor,Delta,phi,iter,eta)
        eig_arr_1m[i,:] = solve_Ham(Ham_eff,num,0.)

    ### Plotting spectra
    for i in range(eig_arr_0.shape[1]):
        if i == 0:
            plt.plot(kx,eig_arr_0[:,i], c= 'k',label = r'$\omega = 0$') # spectrum of H_eff(omega = 0)
            plt.plot(kx,eig_arr_1[:,i], c= 'r',label = r'$\omega = \Delta/3$') # spectrum of H_eff(omega = Delta/3.) # Will be particle-hole asymetric
            plt.plot(kx,eig_arr_1m[:,i], c= 'r',ls = 'dashed',label = r'$\omega = -\Delta/3$') # spectrum of H_eff(omega = -Delta/3.) # Will be particle-hole asymetric
        else:
            plt.plot(kx,eig_arr_0[:,i], c= 'k') # spectrum of H_eff(omega = 0)
            plt.plot(kx,eig_arr_1[:,i], c= 'r') # spectrum of H_eff(omega = Delta/3.) # Will be particle-hole asymetric
            plt.plot(kx,eig_arr_1m[:,i], c= 'r',ls = 'dashed') # spectrum of H_eff(omega = -Delta/3.) # Will be particle-hole asymetric
    plt.grid()
    plt.ylim(-3.*Delta,3.*Delta)
    plt.xlabel(r"$k_x$ $(A^{-1})$",fontsize = 12)
    plt.ylabel(r"$E$ ($meV$)",fontsize = 12)
    plt.legend()
    plt.show()
