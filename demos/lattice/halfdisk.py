import numpy as np
import matplotlib.pyplot as plt

import majoranas.modules.constants as const
import majoranas.modules.lattice as lat
import majoranas.modules.operators as op

R = 20

coor = lat.halfdisk(R)
NN = lat.NN_Arr(coor)
NNk = lat.NN_Bound(NN, coor)

idx = 1
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
