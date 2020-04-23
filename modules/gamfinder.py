import majoranaJJ.operators.sparse.qmsops as spop #sparse operators

def gamfinder(
    coor, ax, ay, NN, mu, NNb = None, Wj = 0, Sx = 0, cutx = 0, cuty = 0, V = 0, gammax = 0, gammay = 0, gammaz = 0, alpha = 0, delta = 0 , phi = 0, qx = 0, qy = 0, periodicX = False, periodicY = False,
    k = 2, sigma = 0, which = 'LM', tol = 0, maxiter = None
    ):

    Ei = spop.EBDG(
        coor, ax, ay, NN, NNb = NNb, Wj = Wj,
        Sx = Sx, cutx = cutx, cuty = cuty,
        V = V, mu = mu,
        gammax = gammax, gammay = gammay, gammaz = gammaz,
        alpha = alpha, delta = delta, phi = phi,
        qx = qx, qy = qy,
        periodicX = periodicX, periodicY = periodicY,
        k = k, sigma = sigma, which = which, tol = tol, maxiter = maxiter
        )[1]

    deltaG = 0.01
    gammanew = gammax + deltaG

    Ef = spop.EBDG(
        coor, ax, ay, NN, NNb = NNb, Wj = Wj,
        Sx = Sx, cutx = cutx, cuty = cuty,
        V = V, mu = mu,
        gammax = gammanew, gammay = gammay, gammaz = gammaz,
        alpha = alpha, delta = delta, phi = phi,
        qx = qx, qy = qy,
        periodicX = periodicX, periodicY = periodicY,
        k = k, sigma = sigma, which = which, tol = tol, maxiter = maxiter
        )[1]

    m = (Ef - Ei)/(gammanew - gammax)
    b = Ei - m*gammax
    G_crit = -b/m

    return G_crit
