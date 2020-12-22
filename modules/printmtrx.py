import majoranaJJ.modules.self_energy as snrg
import majoranaJJ.modules.self_energy_nodule as snrgN
import majoranaJJ.operators.sparse_operators as spop
import majoranaJJ.lattice.nbrs as nb
import majoranaJJ.lattice.shapes as shps
import numpy as np

Wj = 200
ax = 50
ay = 50
Nx = 3
Ny = int(Wj/ay)
#Ny = Ny
Wj_int = Ny

coor = shps.square(3, 1) #square lattice
NN = nb.NN_sqr(coor)
NNb = nb.Bound_Arr(coor)

H1 = spop.HBDG(coor, ax, ay, NN, NNb = NNb, Wj = 0, alpha = 1, delta = 1, phi = np.pi, qx = 1, meff_normal=0.023)
#H2 = snrg.Junc_Ham_gen(Wj, ay, kx=1, m_eff=0.023, alp_l=1, alp_t=1, mu=0,V_J=0,Gam=0)
snrgN.Junc_eff_Ham_gen(
0, Wj, 0, 0, ax, ay, 1, 0.023, 1, 1, 0, 0, 0, delta=1, phi=np.pi)

H1 = H1.todense()
#H2 = H2.todense()
#Junc_eff_Ham_gen(omega, Wj, nodx, nody, ax, ay_targ, kx, m_eff, alp_l, alp_t, mu, Vj, Gam, delta, phi, Gam_SC_factor=0, iter=50, eta=0)
print(H1)
print()
#print(H2)
#for i in range(H1.shape[0]):
#    print(i, H1[i, i], H2[i,i])
