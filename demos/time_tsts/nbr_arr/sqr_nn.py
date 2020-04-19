import time

import majoranaJJ.lattice.shapes as shps #lattice shapes
import majoranaJJ.lattice.nbrs as nb #neighbor arrays
import majoranaJJ.modules.plots as plots #plotting functions

print("")
N = 45

#Making square lattice, nothing has changed with this method
coor = shps.square(N, N)
print("size: ", coor.shape[0])
print("")

###########################################

#General NN_arr method
start = time.time()
NN_gen = nb.NN_Arr(coor)
end = time.time()
print("Time to create Nbr_Arr with general method = {} [s]".format(end-start))

###########################################

start = time.time()
NN_sq = nb.NN_sqr(coor)
end = time.time()
print("Time to create Nbr_Arr with specific square method = {} [s]".format(end-start))

###########################################

#Verifying that the new method creates the same neighbor array as the old one
for i in [0,1,2,3]:
    print("Same Values? ", all(NN_sq[:,i] == NN_gen[:,i]))
