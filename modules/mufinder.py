import scipy.sparse.linalg as spLA
import majoranaJJ.operators.sparsOP as spop
from numpy import sort

def mufinder(
    coor, ax, ay, NN, NNb = None, V = 0, gammax = 0, gammay = 0, gammaz = 0, alpha = 0, qx = 0, qy = 0, periodicX = False, periodicY = False
    ):

    energy = spop.ESOC(coor, ax, ay, NN, NNb=NNb, V=V, mu=0, alpha=alpha, gammax=gammax, gammay=gammay, gammaz=gammaz, qx=qx, qy=qy, periodicX=periodicX, periodicY=periodicY)

    for i in energy:
        if i >= 0.0:
            mu = i
            print("mu = {} (meV)".format(mu))
            return mu

    print("mu = 0 [meV]")
    return 0
