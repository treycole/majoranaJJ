"""
Plotting wavefunctions

def state_cplot(coor, states, title = 'Probability Density'):

    N = coor.shape[0]
    M=states.shape[0]/N
    prob_dens = []
    for i in np.arange(0, N):
        for i in range(0,M+1):
        prob_dens.append(np.square(abs(states[i])) + np.square(abs(states[i+N])))

    print(sum(prob_dens))
    plt.scatter(coor[:,0], coor[:,1], c = prob_dens)
    plt.xlim(0, max(coor[:, 0]))
    plt.ylim(0, max(coor[:, 1]))
    plt.title(title)
    plt.colorbar()
    plt.show()
"""
