import time

import majoranaJJ.lattice.shapes as shp
import majoranaJJ.junk.lattice.neighbors as nb2
import majoranaJJ.lattice.neighbors as nb
import majoranaJJ.etc.plots as plot

print("")
x1 = 20
x2 = 20
y1 = 20
y2 = 20

#Making square lattice, nothing has changed with this method
coor = shp.cross(x1, x2, y1, y2)
print("size: ", coor.shape[0])
print("")

###########################################

#Using old method, scaled by N^2 due to a loop within a loop
start = time.time()
NN_old = nb2.NN_Arr(coor)
end = time.time()
print("Time to create Nbr_Arr with original method = {} [s]".format(end-start))
print(NN_old[0:5, :])

idx = 0
plot.lattice(idx, coor, NN = NN_old)
print(" ")

###########################################

#Using new method, only one loop
start = time.time()
NN_new = nb.NN_Arr(coor)
end = time.time()
print("Time to create Nbr_Arr with revised method = {} [s]".format(end-start))
print(NN_new[0:5, :])

idx = 0
plot.lattice(idx, coor, NN = NN_new)
print(" ")

###########################################

#Verifying that the new method creates the same neighbor array as the old one
for i in [0,1,2,3]:
    print("Same Values? ", all(NN_old[:,i] == NN_new[:,i]))
