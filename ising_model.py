import numpy as np
from numpy.random import default_rng
from scipy.ndimage import convolve, generate_binary_structure
import numba
from numba import njit
import matplotlib as mpl
import matplotlib.pyplot as plt


def gen_random_initial(n=50, n_dim=2, bias=0.5, rng=default_rng()):
    """
    Build randomly filled initial lattice with box length n in n_dim dimensions
    filled with (spins) -1 or 1 with a bias of bias to -1 and a bias of (1 - bias) to 1
    :param n: int box length
    :param n_dim: int number of dimensions (>1, < 4)
    :param bias: bias towards negative spins
    :param rng: np random Generator
    :return: lattice with spins as np.array
    """
    return rng.choice(np.array([-1., 1.], dtype=np.float32),
                      size=tuple(np.repeat(n, repeats=n_dim)),
                      replace=True,
                      p=[bias, 1. - bias])

def make_dist_kernel(arr, r, weigh=False):
    """
    Make a compressed kernel for convolution that includes values within a radius r
    setting weigh -> True weights each pixel with its distance
    :param arr: array with maximum dimensions of the kernel, must be square or cube and have odd box lengths
    :param r: radius at and within which the kernel is set to True
    :param weigh: option to weigh the kernel by distance
    :return: kernel (np.array) of dimension arr
    """
    ny, nx = arr.shape
    assert np.all(np.array(arr.shape) & 0x1) & np.all(np.array(arr.shape) - arr.shape[0] == 0),\
        f"only accept square/cube shapes with odd extents"
    yc, xc = (ny - 1) // 2, (nx - 1) // 2
    yp, xp = np.mgrid[0:ny, 0:nx]
    yp = yp - yc
    xp = xp - xc
    rr = np.sqrt(np.power(yp, 2.) + np.power(xp, 2.))
    kernel = rr <= r
    kernel = weigh * rr * kernel + (1 - weigh) * kernel
    kernel[yc, xc] = 0
    kernel = kernel[~np.all(kernel == 0, axis=1)]
    return kernel[:, ~np.all(kernel == 0, axis=0)]



def get_energy(lattice, kernel):
    """
    Get the energy of a lattice using convolution
    :param lattice: np.array with spin values
    :param kernel: kernel to evaluate the energy
    :return: energy (float)
    """
    arr = -lattice * convolve(lattice, kernel, mode='wrap')
    return arr.sum()


@numba.njit("UniTuple(f8[:], 2)(f8[:,:], i8, f8, f8, f8[:,:])", nopython=True, parallel=True, nogil=True)
def metropolis(lattice, n, bj, energy, kernel):
    """
    Perform metropolis algorithm on lattice in order to find its equilibrium state
    :param lattice:
    :param n:
    :param bj:
    :param energy:
    :param kernel:
    :return:
    """
    # define parameters for rng and index manipulation
    _rng = np.random.default_rng()
    n_grid = np.array(lattice.shape)
    n_kernel = np.array(kernel.shape)

    # define arrays for local lattice and neighbour kernel (stencil)
    stencil = np.nonzero(kernel)
    _lattice = np.copy(lattice)

    # define arrays to keep track of the metropolis algorithm
    net_spins = np.zeros(n - 1)
    net_energy = np.zeros(n - 1)

    # propagate over n states
    for t in np.arange(n - 1):
        # select random element
        # TODO: adjust for potential 3d
        x = np.random.randint(n_grid[0])
        y = np.random.randint(n_grid[1])

        # get initial spin and define trial spin (switch state)
        initial_spin = _lattice[x, y]
        trial_spin = initial_spin * -1

        # get the spin of the neighbours that are influenced by the switch
        offsets = np.array([x, y]) - n_kernel // 2
        _stencil = list([idxs + offset for idxs, offset in zip(stencil, offsets)])
        __stencil = np.ravel_multi_index(_stencil, _lattice.shape, mode="wrap")
        neighbour_spins = np.take(_lattice, __stencil)

        # find the energies of the initial and trial sub-lattice state
        initial_energy = np.sum(initial_spin * neighbour_spins)
        trial_energy = np.sum(trial_spin * neighbour_spins)

        # get energy change due to trial
        delta_energy = trial_energy - initial_energy

        # amazing magic that does not require comparisons (kinda proud of that one)
        # 1) return 1 if dE >= 0 return 0 otherwise
        # 2) return 1 if dE >= 0,
        #                           (return 1 if RNG[0, 1) < np.exp(-beta * J * dE), return 0 otherwise)
        # 3) change lattice state if 1, keep lattice state if 0
        # 4) add dE to E if 1
        # neat, right?

        prob = np.clip(delta_energy + 1, a_min=0., a_max=1.).astype(int)
        prob = (np.clip(np.random.random() - np.exp(-bj * delta_energy) + 1., a_min=0., a_max=1.).astype(int) + (1 - prob))

        _lattice[x, y] = trial_spin * prob + _lattice[x, y] * (1 - prob)

        energy += delta_energy * prob

        net_spins[t] = _lattice.sum()
        net_energy[t] = energy

    return net_spins, net_energy


def run_ising(n=50, n_dim=2):
    return


if __name__ == "__main__":
    arr = np.zeros((11, 11))
    kernel = make_dist_kernel(arr, 2., )

    test = np.arange(11*11).reshape((11, 11))

    plt.matshow(test)
    plt.show()

    y, x = 9, 3


    print(kernel)

    n_kernel = np.array(kernel.shape)
    stencil = np.nonzero(kernel) # - n_kernel // 2

    offsets = np.array([x, y]) - n_kernel // 2
    print(offsets)

    # stencil = tuple([idxs + offset for idxs, offset in zip(stencil, n_kernel // 2 - 1)])

    stencil = tuple([idxs + offset for idxs, offset in zip(stencil, offsets)])

    print(stencil)

    stencil = np.ravel_multi_index(stencil, test.shape, mode="wrap")

    print(stencil)

    test_ver = np.copy(test)
    test_ver[x, y] = 200
    plt.matshow(test_ver)
    plt.show()

    np.put(test, stencil, v=150)

    plt.matshow(test)
    plt.show()

    # side_length = 25
    # number_dimensions = 2
    # run_ising(side_length, number_dimensions)