import matplotlib.pyplot as plt

width = 3*3.375
height = width/1.2

params = {
    'axes.labelsize': 12,
    'axes.linewidth': 0.7,
    'font.size': 12,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': (width, height),
}

plt.rcParams.update(params)
