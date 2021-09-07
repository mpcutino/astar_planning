import numpy as np
import matplotlib.pyplot as plt


def distance(state1, state2):
    return np.sum((fs2numpy(state1) - fs2numpy(state2))**2)


def fs2numpy(state):
    return np.array([state.x, state.z])


def plot_from_fs(fs_path, initial=None, target=None):
    """
    plot path from a given list of Flight States.
    If initial is None, then the first state is assumed to be the starting one.
    """
    if initial is not None:
        fs_path = [initial] + fs_path

    xs = [i.x for i in fs_path]
    ys = [i.z for i in fs_path]

    plt.plot(xs, ys, marker='.')
    plt.scatter(fs_path[0].x, fs_path[0].z, c='yellow')
    plt.scatter(target.x, target.z, c='green')
    plt.text(fs_path[-1].x-0.5, fs_path[-1].z-0.2, "cost: {0:.2f}".format(fs_path[-1].cost))

    plt.show()
    plt.close()
