from ising_model import *

def kernel():
    arr = np.zeros((11, 11))
    kernel = make_dist_kernel(arr, 1., )

    test = np.arange(11 * 11).reshape((11, 11))
    n_grid = np.array(test.shape)

    plt.matshow(test)
    plt.show()

    plt.matshow(kernel)
    plt.show()

    y, x = 10, 3

    test[x, y] = 200

    n_kernel = np.array(kernel.shape)
    stencil = np.nonzero(kernel)  # - n_kernel // 2

    offsets = np.array([x, y]) - n_kernel // 2
    stencil = tuple([idxs + offset for idxs, offset in zip(stencil, offsets)])
    print(stencil)
    stencil = np.ravel_multi_index(stencil, test.shape, mode="wrap")
    print(stencil)

    np.put(test, stencil, v=150)
    plt.matshow(test)
    plt.show()
    neighbour_spins = np.take(test, stencil)
    print(neighbour_spins)

    stencil = np.nonzero(kernel)
    offsets = np.array([x, y]) - n_kernel // 2
    _stencil = [idxs + offset for idxs, offset in zip(stencil, offsets)]

    print(_stencil)

    __stencil = [idxs % max_idx for idxs, max_idx in zip(_stencil, n_grid)]

    __stencil = __stencil[0] * n_grid[0] + __stencil[1]

    print(__stencil)

    # __stencil = (_stencil[0] % (n_grid[0] - 1)) * n_grid[0] + (_stencil[1] % (n_grid[1] - 1)) * n_grid[1]
    neighbour_spins = np.take(test, __stencil)

    print(neighbour_spins)















if __name__ == "__main__":
    kernel()
