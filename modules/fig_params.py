import matplotlib.pyplot as plt

width = (3+3/8)
height = width*0.95/2

params = {
    'axes.labelsize': 12,
    'axes.linewidth': 2,
    'font.size': 10,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': (width, height),
}

plt.rcParams.update(params)

"""
#Straight junction system parameters (FIG 1 and FIG 2)
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 3 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
cutx = 0 #width of nodule
cuty = 0 #height of nodule
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
delta = 0.3 #Superconducting Gap: [meV]

    #Phase Boundary
    phi = [0, pi]
    Vj = 0 #junction potential: [meV]
    mu_i = -2
    mu_f = 20
    res = 0.01
    gi = 0
    gf = 5
    num_bound = 4
    #Gap vs mu
    phi = [0, pi]
    Vj = 0
    gx = 1
    mu_i = -2
    mu_f = 12
    #Gap vs gam
    phi = [0, pi]
    mu = [0, 10] #for phi = 0, pi respectively
    Vj = 0
    gi = 0
    gf = 3
    #Gap vs #Gap vs gamj
    phi = np.pi
    mu = 10
    gx = 1
    Vj_i = -11
    Vj_f = 11


#Junction with nodule system parameters (FIG 4, 5, 6)
ax = 50 #lattice spacing in x-direction: [A]
ay = 50 #lattice spacing in y-direction: [A]
Nx = 12 #Number of lattice sites along x-direction
Wj = 1000 #Junction region [A]
cutx = 4 #width of nodule
cuty = 8 #height of nodule
alpha = 200 #Spin-Orbit Coupling constant: [meV*A]
delta = 0.3 #Superconducting Gap: [meV]

    #Phase boundary
    phi = [0,np.pi] #SC phase difference
    Vj = -40 #junction potential: [meV]
    mu_i = -5
    mu_f = 15
    res = 0.005
    gi = 0
    gf = 5.0
    num_bound = 10

    #Gap vs mu
    Vj = -40
    phi = [0, pi]
    gx = 1 #mev
    mu_i = -2
    mu_f = 12
    delta_mu = mu_f - mu_i
    res = 0.01
"""
