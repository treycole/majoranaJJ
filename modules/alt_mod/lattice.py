#defining lattice, numbered 0->N
def lattice(Nx, Ny):
    lattice = np.zeros((Nx, Ny))
    for i in range(Ny):
        for j in range(Nx):
            lattice[i, j] = j + i*Ny
    return lattice
