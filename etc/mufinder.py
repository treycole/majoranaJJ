import scipy.sparse.linalg as spLA
import majoranaJJ.operators.sparsOP as spop
from numpy import sort

def mufinder(
    coor, ax, ay, NN, NNb = None, V = 0, gammax = 0, gammay = 0, gammaz = 0, alpha = 0, periodicX = True, periodicY = True
    ):

    Energy = spop.ESOC(coor, ax, ay, NN, NNb=NNb, V=V, alpha=alpha, gammaz=gammaz, periodicX=True, periodicY=True)

    for i in Energy:
        if i >= 0.0:
            mu = i
            print("mu = ", mu*1000,  "[meV]")
            return mu

    print("mu = 0 [meV]")
    return 0
