def Hjunc(
Wj, nodx, nody, ay, ax, kx, m_eff, alp_l, alp_t, mu, Vj, gamx
):
    # Generates the BdG Hamiltonian of the isolated junction region (i.e. no SC included).
    #    * W is the width of the junction
    #    * ay_targ is the targeted lattice constant
    #    * kx is the wavevector along the length of the junction
    #    * m_eff is the effective mass
    #    * alp_l is the longitudinal spin-orbit coupling coefficient
    #    * alp_t is the transverse spin-orbit coupling coefficient
    #    * mu is the chemical potential
    #    * V_J is an addition potential in the junction region (V = 0 in SC regions by convention)
    #     * Gam is the Zeeman energy

    N = int(W/ay_targ) - 1 # number of lattice sites in the junction (in the y-direction)
    ay = W/float(N+1) # actual lattice constant

    t = -1000.*par.hbm0/(2.*m_eff*ay**2) # spin-preserving hopping strength
    t_alp = alp_t / (2.*ay)     # spin-orbit hopping strength in the y-direction
    alp_onsite = kx * alp_l    # onsite spin-orbit coupling contribution
    Tx = 1000.*par.hbm0 * kx**2/(2.*m_eff) # kinetic energy from momentum in x-direction

    row = []; col = []; data = []
    for i in range(N):
        # onsite terms
        row.append(i + 0); col.append(i + 0); data.append(-2*t - mu + Tx + Vj)
        row.append(i + N); col.append(i + N); data.append(-2*t - mu + Tx + Vj)
        row.append(i + 0); col.append(i + N); data.append(-1j*alp_onsite + Gam)
        row.append(i + N); col.append(i + 0); data.append(1j*alp_onsite + Gam)

        row.append(i + 2*N); col.append(i + 2*N); data.append(-(-2*t - mu + Tx + Vj))
        row.append(i + 3*N); col.append(i + 3*N); data.append(-(-2*t - mu + Tx + Vj))
        row.append(i + 2*N); col.append(i + 3*N); data.append(1j*alp_onsite - Gam)
        row.append(i + 3*N); col.append(i + 2*N); data.append(-1j*alp_onsite - Gam)

        # nearest neighbor terms
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

def top_SC_sNRG_calc(
omega, Wj, nodx, nody, ay, ax, kx, m_eff, alp_l, alp_t, mu, Gam_SC, delta, iter, eta):
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
        Nx = 1
    else:
        Nx = nodx+2
    Ny = int(Wj/ay_targ) - 1 # number of lattice sites in the junction region (in the y-direction)
    ay = Wj/float(Ny+1)      # actual lattice constant

    tx = -1000.*par.hbm0/(2.*m_eff*ax**2)  # spin-preserving hopping strength
    ty = -1000.*par.hbm0/(2.*m_eff*ay**2) # spin-preserving hopping strength
    ty_alp = alp_t / (2.*ay) # spin-orbit hopping strength in the y-direction
    tx_alp = alp_l/(2*ax)  # onsite spin-orbit coupling contribution
    ep_on = -2*tx - 2*ty - mu
    dc = np.conjugate(delta)

    ### Onsite Hamiltonian matrix
    H00 = np.zeros((4*Nx, 4*Nx))
    H10 = np.zeros((4*Nx, 4*Nx))
    H01 = np.zeros((4*Nx, 4*Nx))
    Delta = np.zeros((2*Nx, 2*Nx))

    for i in range(Nx):
        H10[i, i] = ty
        H10[i+Nx, i+Nx] = ty
        H10[i+2*Nx, i+2*Nx] = -ty
        H10[i+3*Nx, i+3*Nx] = -ty
        H10[i, i+Nx] = -1j*ty_alp
        H10[i+Nx, i] = -1j*ty_alp
        H10[i+2*Nx, i+Nx+2*Nx] = -1j*ty_alp
        H10[i+Nx+2*Nx, i+2*Nx] = -1j*ty_alp

        H00[i, i] = ep_on
        H00[i+Nx, i+Nx] = ep_on
        H00[i+2*Nx, i+2*Nx] = -ep_on
        H00[i+3*Nx, i+3*Nx] = -ep_on

        H00[i, i+Nx+2*Nx] = delta
        H00[i+Nx, i+2*Nx] = -delta
        H00[i+2*Nx, i+Nx] = -delta
        H00[i+Nx+2*Nx, i+Nx] = delta
        if i != Nx-1:
            H00[i, i+1] = tx
            H00[i+1, i] = tx

            H00[i+1, i+Nx] = -tx_alp
            H00[i+Nx, i+1] = -tx_alp
            H00[i, i+Nx+1] = tx_alp
            H00[i+Nx+1, i] = tx_alp
            H00[i, i+Nx+1] = tx_alp
            H00[i+Nx+1, i] = tx_alp
            H00[i+1, i+Nx] = -tx_alp
            H00[i+Nx, i+1] = -tx_alp

            H00[i+Nx, i+Nx+1] = tx
            H00[i+Nx+1, i+Nx] = tx

            H00[i+2*Nx, i+2*Nx+1] = -tx
            H00[i+2*Nx+1, i+2*Nx] = -tx

            H00[i+3*Nx, i+3*Nx+1] = -tx
            H00[i+3*Nx+1, i+3*Nx] = -tx

        if i == Nx-1:
            #particle
            H00[i, 0] = tx*np.exp(1j*kx*ax)
            H00[0, i] = tx*np.exp(-1j*kx*ax)
            H00[i+Nx, 0+Nx] = tx*np.exp(1j*kx*ax)
            H00[0+Nx, i+Nx] = tx*np.exp(-1j*kx*ax)

            H00[i, 0+Nx] = -tx_alp*np.exp(1j*kx*ax)
            H00[0+Nx, i] = -tx_alp*np.exp(-1j*kx*ax)
            H00[i+Nx, 0 ] = tx_alp*np.exp(1j*kx*ax)
            H00[0, i+Nx] = tx_alp*np.exp(-1j*kx*ax)

        #hole
        H00[i+2*Nx, 0+2*Nx] = np.conjugate(-tx*np.exp(-1j*kx*ax))
        H00[0+2*Nx, i+2*Nx] = np.conjugate(-tx*np.exp(1j*kx*ax)
        H00[i+Nx+2*Nx, 0+Nx+2*Nx] = np.conjugate(-tx*np.exp(-1j*kx*ax))
        H00[0+Nx+2*Nx, i+Nx+2*Nx] = np.conjugate(-tx*np.exp(1j*kx*ax))

        H00[i+2*Nx, 0+Nx+2*Nx] = np.conjugate(tx_alp*np.exp(-1j*kx*ax))
        H00[0+Nx+2*Nx, i+2*Nx] = np.conjugate(tx_alp*np.exp(1j*kx*ax))
        H00[i+Nx+2*Nx, 0+2*Nx ] = np.conjugate(-tx_alp*np.exp(-1j*kx*ax))
        H00[0+2*Nx, i+Nx+2*Nx] = np.conjugate(-tx_alp*np.exp(1j*kx*ax))

    H01 = np.conjugate(np.transpose(H10))
    ### Identity matrix
    I = np.eye(4*Nx, dtype = 'complex')
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
    for m in range(4):
        for n in range(4):
            for i in range(Nx):
                for j in range(Ny):
                    row.append((Ny-1)*Nx+i + m*Nx*Ny); col.append((Ny-1)*Nx+j + n*Nx*Ny); data.append(sNRG[m*Nx+i,n*Nx+j])
    sNRG_mtx = Spar.csc_matrix((data,(row,col)), shape = (4*N,4*N), dtype = 'complex')

    return G_s, G_b, sNRG_mtx

def bot_SC_sNRG_calc(omega,W,ay_targ,kx,m_eff,alp_l,alp_t,mu,Gam_SC,Delta,iter,eta):
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
    #     * Delta is the (complex) SC order parameter in the top SC
    #     * iter is the number of iteration of the algorithm to perform
    #     * eta is the imaginary component of the energy that is used for broadening

    if nodx == 0:
        Nx = 1
    else:
        Nx = nodx+2
    Ny = int(Wj/ay_targ) - 1 # number of lattice sites in the junction region (in the y-direction)
    ay = Wj/float(Ny+1)      # actual lattice constant

    tx = -1000.*par.hbm0/(2.*m_eff*ax**2)  # spin-preserving hopping strength
    ty = -1000.*par.hbm0/(2.*m_eff*ay**2) # spin-preserving hopping strength
    ty_alp = alp_t / (2.*ay) # spin-orbit hopping strength in the y-direction
    tx_alp = alp_l/(2*ax)  # onsite spin-orbit coupling contribution
    ep_on = -2*tx - 2*ty - mu
    dc = np.conjugate(delta)

    ### Onsite Hamiltonian matrix
    H00 = np.zeros((4*Nx, 4*Nx))
    H10 = np.zeros((4*Nx, 4*Nx))
    H01 = np.zeros((4*Nx, 4*Nx))
    Delta = np.zeros((2*Nx, 2*Nx))

    for i in range(Nx):
        H10[i, i] = ty
        H10[i+Nx, i+Nx] = ty
        H10[i+2*Nx, i+2*Nx] = -ty #-H0(-k)*
        H10[i+3*Nx, i+3*Nx] = -ty #-H0(-k)*
        H10[i, i+Nx] = -1j*ty_alp*-1 #bottom SC, negative sign
        H10[i+Nx, i] = -1j*ty_alp*-1 #bottom SC, negative sign
        H10[i+2*Nx, i+Nx+2*Nx] = -1j*ty_alp*-1 #bottom SC, negative sign
        H10[i+Nx+2*Nx, i+2*Nx] = -1j*ty_alp*-1 #bottom SC, negative sign

        H00[i, i] = ep_on
        H00[i+Nx, i+Nx] = ep_on
        H00[i+2*Nx, i+2*Nx] = -ep_on
        H00[i+3*Nx, i+3*Nx] = -ep_on

        H00[i, i+Nx+2*Nx] = dc #delta conjugate
        H00[i+Nx, i+2*Nx] = -dc
        H00[i+2*Nx, i+Nx] = -dc
        H00[i+Nx+2*Nx, i+Nx] = dc
        if i != Nx-1:
            H00[i, i+1] = tx
            H00[i+1, i] = tx

            H00[i+1, i+Nx] = -tx_alp
            H00[i+Nx, i+1] = -tx_alp
            H00[i, i+Nx+1] = tx_alp
            H00[i+Nx+1, i] = tx_alp
            H00[i, i+Nx+1] = tx_alp
            H00[i+Nx+1, i] = tx_alp
            H00[i+1, i+Nx] = -tx_alp
            H00[i+Nx, i+1] = -tx_alp

            H00[i+Nx, i+Nx+1] = tx
            H00[i+Nx+1, i+Nx] = tx

            H00[i+2*Nx, i+2*Nx+1] = -tx
            H00[i+2*Nx+1, i+2*Nx] = -tx

            H00[i+3*Nx, i+3*Nx+1] = -tx
            H00[i+3*Nx+1, i+3*Nx] = -tx

        if i == Nx-1:
            #particle
            H00[i, 0] = tx*np.exp(1j*kx*ax)
            H00[0, i] = tx*np.exp(-1j*kx*ax)
            H00[i+Nx, 0+Nx] = tx*np.exp(1j*kx*ax)
            H00[0+Nx, i+Nx] = tx*np.exp(-1j*kx*ax)

            H00[i, 0+Nx] = -tx_alp*np.exp(1j*kx*ax)
            H00[0+Nx, i] = -tx_alp*np.exp(-1j*kx*ax)
            H00[i+Nx, 0 ] = tx_alp*np.exp(1j*kx*ax)
            H00[0, i+Nx] = tx_alp*np.exp(-1j*kx*ax)

            #hole
            H00[i+2*Nx, 0+2*Nx] = np.conjugate(-tx*np.exp(-1j*kx*ax))
            H00[0+2*Nx, i+2*Nx] = np.conjugate(-tx*np.exp(1j*kx*ax)
            H00[i+Nx+2*Nx, 0+Nx+2*Nx] = np.conjugate(-tx*np.exp(-1j*kx*ax))
            H00[0+Nx+2*Nx, i+Nx+2*Nx] = np.conjugate(-tx*np.exp(1j*kx*ax))

            H00[i+2*Nx, 0+Nx+2*Nx] = np.conjugate(tx_alp*np.exp(-1j*kx*ax))
            H00[0+Nx+2*Nx, i+2*Nx] = np.conjugate(tx_alp*np.exp(1j*kx*ax))
            H00[i+Nx+2*Nx, 0+2*Nx ] = np.conjugate(-tx_alp*np.exp(-1j*kx*ax))
            H00[0+2*Nx, i+Nx+2*Nx] = np.conjugate(-tx_alp*np.exp(1j*kx*ax))

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
    for m in range(4):
        for n in range(4):
            for i in range(Nx):
                for j in range(Ny):
                    row.append((Ny-1)*Nx+i + m*Nx*Ny); col.append((Ny-1)*Nx+j + n*Nx*Ny); data.append(sNRG[m*Nx+i,n*Nx+j])
    sNRG_mtx = Spar.csc_matrix((data,(row,col)), shape = (4*N,4*N), dtype = 'complex')

    return G_s, G_b, sNRG_mtx
