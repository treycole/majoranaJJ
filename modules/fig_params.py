import matplotlib.pyplot as plt

width = (3+3/8)
height = width*0.8

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
