import time

import majoranaJJ.lattice.shapes as shp
import majoranaJJ.etc.neighbors as nb2
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.plots as plot

print("")
R = 30
r = 5

#Making square lattice, nothing has changed with this method
coor = shp.donut(R, r)
print("size: ", coor.shape[0])
print("")

###############################################################################

#Using old method, scaled by N^2 due to a loop within a loop
start = time.time()
NN_old = nb2.NN_Arr(coor)
end = time.time()
print("time for original method = ", end-start)
print(NN_old)

idx = 0
plot.neyb(idx, coor, NN = NN_old)
print(" ")

###############################################################################

start = time.time()
NN_new = nb.NN_Arr(coor)
end = time.time()
print("time for new method = {} [s]".format(end-start))
print(NN_new)

idx = 0
plot.neyb(idx, coor, NN = NN_new)
print(" ")

###############################################################################

#Verifying that the new method creates the same neighbor array as the old one
NN_new = nb.NN_Arr(coor)
for i in [0,1,2,3]:
    print("Same Values? ", all(NN_old[:,i] == NN_new[:,i]))
