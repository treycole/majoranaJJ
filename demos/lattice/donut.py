import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.lines as mlines
from scipy import interpolate

import majoranas.modules.constants as const
import majoranas.modules.lattice as lat
import majoranas.modules.operators as op

R = 25
r = 10

coor = lat.donut(R, r) #donut coordinate array
NN = lat.NN_Arr(coor) #nearest neighbor array for donut
NNk = lat.NN_Bound(NN, coor) #nerest neighbor boundary

idx = 1
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
