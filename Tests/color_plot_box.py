#Only to test method of color plotting, it is however the particle in a box wavefunction
import numpy as np

N_x = 100
N_y = 100

test = np.zeros((N_x, N_y))

x = np.linspace(0, np.pi, N_x)
y = np.linspace(0, np.pi, N_y)
for i in range(N_x):
    for j in range(N_y):
        test[i, j] = np.sin(9*x[i])**2*np.sin(2*y[j])**2
#print(test)
plt.pcolormesh(test)
plt.show()
