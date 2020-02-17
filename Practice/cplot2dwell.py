# Colormesh plot for amplitude of wave function for two dimensional potential well
import numpy as np
import matplotlib.pyplot as plt

nx = 2
ny = 2
x= np.linspace(-1*np.pi/2, 1*np.pi/2,500)
y= np.linspace(-1*np.pi/2, 1*np.pi/2,500)

x,y = np.meshgrid(x,y)
z = np.sin(nx*x)**2*np.sin(ny*y)**2 # amplitude of wavefunction for two dimensional box

plt.pcolormesh(x,y,z)
plt.show()
