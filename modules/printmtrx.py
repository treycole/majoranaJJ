import majoranaJJ.modules.self_energy as snrg
import majoranaJJ.modules.self_energy_nodule as snrgN
import majoranaJJ.operators.sparse_operators as spop
import majoranaJJ.lattice.nbrs as nb
import majoranaJJ.lattice.shapes as shps
import numpy as np

Wj = 100
ax = 50
ay = 50
Nx = 3
Ny = int(Wj/ay)
#Ny = Ny
Wj_int = Ny

coor = shps.square(Nx, Ny) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)

print((.1/Wj)*(Nx*ax), "?")
H1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = Wj_int, alpha = 100, delta = 1, phi = np.pi, qx=(.1/Wj), meff_normal=0.023)
#H2 = snrg.Junc_Ham_gen(Wj, ay, kx=0, m_eff=0.023, alp_l=100, alp_t=100, mu=0,V_J=0,Gam=0)
#(omega, Wj, Lx, nodx, nody, ax, ay_targ, kx, m_eff, alp_l, alp_t, mu, Vj, Gam, delta, phi, Gam_SC_factor=0, iter=50, eta=0)
snrgN.Junc_eff_Ham_gen(0, Wj, (Nx*ax), 0, 0, ax, ay, (.1/Wj), 0.023, 100, 100, 0, 0, 0, delta=1, phi=np.pi)
print()
H1 = H1.todense()
#H2 = H2.todense()
#Junc_eff_Ham_gen(omega, Wj, nodx, nody, ax, ay_targ, kx, m_eff, alp_l, alp_t, mu, Vj, Gam, delta, phi, Gam_SC_factor=0, iter=50, eta=0)
print(np.around(H1.real, decimals=2))
print()
#print(np.around(H2.imag, decimals=1))
#for i in range(H1.shape[0]):
#    print(i, H1[i, i], H2[i,i])
