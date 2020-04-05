import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import matplotlib.lines as mlines
from scipy import interpolate

import majoranas.modules.constants as const
import majoranas.modules.lattice as lat
import majoranas.modules.operators as op

Ny = 25    #number of lattice sites in y direction
Nx = 25     #number of lattice sites in x direction
N = Ny*Nx

xbase = 40
xcut = 5
y1 = 10
y2 = 10

coor = lat.Ibeam(xbase, xcut, y1, y2)
NN = lat.NN_Arr(coor)
NNk = lat.NN_Bound(NN, coor)

idx = 1
plots.lattice(idx, coor, NN = NN)
plots.lattice(idx, coor, NNb = NNb)
