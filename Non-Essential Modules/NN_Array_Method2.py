#alternative algorithm to get NN array. Uses defined lattice sites for conditional statements, may be useful when changing
# .. geometry

count = 0
NN2 = np.zeros((N,4))
for i in range(len(lattice[0])):
    for j in range(len(lattice[1])):
        if j == 0:
            NN[count, 0] = -1
        else:
            NN[count, 0] = lattice[i,j] - 1
        if i == 0:
            NN[count, 1] = -1
        else:
            NN[count, 1] = lattice[i,j] - Nx
        if j == Nx-1:
            NN[count, 2] = -1
        else:
            NN[count, 2] = lattice[i,j] + 1
        if i == Nx-1:
            NN[count, 3] = -1
        else:
            NN[count, 3] = lattice[i,j] + Nx
        count += 1

#for i in (NN == np.transpose(NN2)):
#    if (i.any())!= True:
#        print('Not equal')
