import matplotlib.pyplot as plt
        ### Physical Constants ###
hbar =6.58211899*10**(-16)                  # h/2Pi  in eV s *)
m0=9.10938215*10**(-31)
e0=1.602176487*10**(-19)
hbm0 = (hbar)**(2) * e0 * 10**(20)/m0       # hbar^2 / m0 in eV * A^2
kb = 8.61733 * 10**(-5) * 1000              # boltzmann constant in meV / K
ep_knot = 8.85 * 10**(-22) # C / (V*A) where A is an Angstrom
phi_knot = 2.067833848 * 10**(5)            # magnetic flux quantum in T * A^2
muB = 5.7883818012 * 10**(-5)       # Bohr magneton in eV * T^-1

width = 3.847 * 2.
height = width/ (1.62)     #The golden ratio
params = {
   'axes.labelsize': 10,
   'axes.linewidth': 1.7,
   'font.size': 10,
   'legend.fontsize': 10,
   'xtick.labelsize': 12,
   'ytick.labelsize': 12,
   'text.usetex': False,
   'figure.figsize': (width,height),
   }
plt.rcParams.update(params)
