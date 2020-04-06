import time
import matplotlib.pyplot as plt

import majoranaJJ.lattice.shapes as shp
import majoranaJJ.junk.lattice.neighbors as nb2
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.etc.plots as plot

print(" ")
R = 30
r = 5

#Making square lattice, nothing has changed with this method
#Finding neighbor array in the unit cell for the old method to find boundary array.
coor = shp.donut(R, r)
NN = nb.NN_Arr(coor)
print("size: ", coor.shape[0])
print("")

##########################################

#Using old method, scaled by N^2 due to a loop within a loop
start = time.time()
NNb2 = nb2.NN_Bound(NN, coor)
end = time.time()
print("time for original method = ", end-start)
print(NNb2[0:5, :])

idx = 0
plot.lattice(idx, coor, NNb = NNb2)
print(" ")

###########################################

start = time.time()
NNb = nb.NN_Bound(coor)
end = time.time()
print("Time to create Boundary array, implemented using NumPy with revised algorithms", end-start)
print(NNb[0:5, :])

idx = 0
plot.lattice(idx, coor, NNb = NNb)
print(" ")

##########################################

#Verifying that the new method creates the same neighbor array as the old one
for i in [0,1,2,3]:
    print("Same Values? ", all(NNb[:,i] == NNb2[:,i]))
