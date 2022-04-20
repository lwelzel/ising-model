import numpy as np
from numpy.random import default_rng
from scipy.ndimage import convolve
import numba
from numba import njit
from statsmodels.tsa.stattools import acf
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm
import h5py
from pathlib import Path
from time import strftime, gmtime

#for testing
from scipy.signal import savgol_filter


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
    return rng.choice(np.array([-1., 1.],
                               dtype=np.float32, order="C"),
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
    assert np.all(np.array(arr.shape) & 0x1) & np.all(np.array(arr.shape) - arr.shape[0] == 0), \
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
    return kernel[:, ~np.all(kernel == 0, axis=0)].astype("float32", order="C")


def get_energy(lattice, kernel):
    """
    Get the energy of a lattice using convolution
    :param lattice: np.array with spin values
    :param kernel: kernel to evaluate the energy
    :return: energy (float)
    """
    arr = -lattice * convolve(lattice, kernel, mode='wrap')
    return float(arr.sum() / 2.)


@njit(parallel=False, nogil=True, fastmath=True)
def get_autocorr_time_lectures(mag):
    """
    Find autocorrelation time using the method shown in the lecture.
    This is slow and not very cool, sorry :/
    :param mag:
    :return:
    """
    auto_corr_func = np.zeros_like(mag)
    for t in np.arange(mag.size):
        t_prime_max = auto_corr_func.size - t
        fact = 1 / (t_prime_max)
        auto_corr_func[t] = fact * np.sum(mag[:t_prime_max] * mag[t:t_prime_max + t])
        auto_corr_func[t] += - fact * (np.sum(mag[:t_prime_max]) * np.sum(mag[t:t_prime_max + t]))

    auto_corr_func = auto_corr_func / auto_corr_func[0]
    flip_idx = np.argmin(np.sign(auto_corr_func))
    auto_corr_func = auto_corr_func[:flip_idx]
    tau = np.trapz(auto_corr_func)
    return tau

def get_autocorr_time(mag):
    """
    Calculate autocorrelation time using FFT
    :param mag:
    :return:
    """
    # TODO: also check https://dfm.io/posts/autocorr/
    # compute acf via FFT
    # acf is normalized
    nlags = int(np.minimum(mag.size * 0.95, 1e5))
    auto_corr_func, conf = acf(mag, nlags=nlags, alpha=0.05)

    # add 1 for safety but is probably not really needed since the relative difference is so small
    flip_idx = np.argmin(np.sign(auto_corr_func)) + 1
    if np.logical_or(flip_idx == 0, flip_idx == 1):
        flip_idx = mag.size
    tau = np.trapz(auto_corr_func[:flip_idx])
    # # TODO: take out visualization
    # plot_acf(mag, lags=nlags, alpha=0.05)
    # plt.axvline(tau, c="k")
    # plt.show()
    return tau


def get_std_therm_av(m, tau):
    return np.sqrt(2 * tau / m.size * (np.std(np.square(m)) - np.square(np.std(m))))


# TODO: option parallel=True sadly doesnt work at the moment,
#  issue is with how numba handles (or rather doesnt) broadcasting.
#  See: https://github.com/numba/numba/issues/3729
# TODO: this leaks memory like a sponge, no idea why.
#  See:
#  Option 1: https://github.com/numba/numba/issues/4299
#  Option 2: https://numba.pydata.org/numba-doc/dev/developer/numba-runtime.html
# TODO: use explicit typing?
#  "UniTuple(f8[:],2)(f8[:,:], f8[:,:], i8, f8, f8)"
@njit(parallel=False, nogil=True, fastmath=True)
def metropolis(lattice, kernel, n, bj, energy):
    """
    Perform metropolis algorithm on lattice in order to find its equilibrium state.
    This is amazing code and deserves an extra point.
    :param lattice:
    :param n:
    :param bj:
    :param energy:
    :param kernel:
    :return:
    """

    # define parameters for rng and index manipulation
    n_grid = np.array(lattice.shape)
    n_kernel = np.array(kernel.shape)

    # define arrays for local lattice and neighbour kernel (stencil)
    stencil = np.nonzero(kernel)
    # redefine for numba
    stencil = np.array([
        [*stencil[0]],
        [*stencil[1]]
    ])
    _lattice = np.copy(lattice)

    # define arrays to keep track of the metropolis algorithm
    system_magnetisation_tracker = np.zeros(n - 1)
    system_energy_tracker = np.zeros(n - 1)

    mc_sweep = np.arange(np.prod(n_grid))

    # propagate over n MC sweeps
    for t in np.arange(n - 1):
        for trans in mc_sweep:
            # select random element
            # TODO: adjust for potential 3d
            x = np.random.randint(n_grid[0])
            y = np.random.randint(n_grid[1])

            # get initial spin and define trial spin (switch state)
            initial_spin = _lattice[x, y]
            trial_spin = -initial_spin

            # get the spin of the neighbours that are influenced by the switch
            # its a bit messy due to the way numba wants stuff
            offsets = np.array([x, y]) - n_kernel // 2
            _stencil = np.add(stencil, offsets.reshape((-1, 1)))
            wrapped_stencil = _stencil % n_grid.reshape((-1, 1))
            flat_stencil = wrapped_stencil[0] * n_grid[0] + wrapped_stencil[1]
            neighbour_spins = np.take(_lattice, flat_stencil)

            # find the energies of the initial and trial sub-lattice state
            initial_energy = np.sum(- initial_spin * neighbour_spins)
            trial_energy = np.sum(- trial_spin * neighbour_spins)

            # get energy change due to trial
            delta_energy = trial_energy - initial_energy

            # print(trial_energy, " - " ,initial_energy, " = ", delta_energy)

            # amazing magic that does not require comparisons (kinda proud of that one)
            # 1) return 1 if dE >= 0 return 0 otherwise
            # 2) return 1 if dE >= 0,
            #                           (return 1 if RNG[0, 1) < np.exp(-beta * J * dE), return 0 otherwise)
            # 3) change lattice state if 1, keep lattice state if 0
            # 4) add dE to E if 1
            # neat, right?
            prob = int(np.minimum(1., np.maximum(np.exp(-bj * delta_energy) + 1. - np.random.random(), 0.)))

            # update system
            _lattice[x, y] = trial_spin * prob + _lattice[x, y] * (1 - prob)
            energy += delta_energy * prob

        # save results
        system_magnetisation_tracker[t] = np.sum(_lattice)
        system_energy_tracker[t] = energy

    return system_magnetisation_tracker, system_energy_tracker, _lattice


def run_ising(n=50, n_dim=2, n_runs=int(1e3), bj=0.25, bias=0.75, kernel_radius=1.,
              _id=0, disable_relaxation=False):

    assert n_runs >= int(1e2), "the relaxation must have a sufficient length"

    ### INITIALIZE
    lattice = gen_random_initial(n, n_dim, bias)
    kernel = make_dist_kernel(lattice[:-(1 - n % 2), :-(1 - n % 2)], r=kernel_radius)
    initial_energy = get_energy(lattice, kernel)
    relaxation_tol = 0.1
    relaxation_range = int(0.25 * n_runs)
    max_relaxation_iters = 25

    ### RELAX
    # while it looks like we have not settled into a state which we know is relaxed continue relaxing
    # states that I assume are relaxed
    # - mean magnetisation is close to 1 or -1
    # - mean magnetisation is close to 0
    # - mean magnetisation quickly switches between 1 and -1 but is close to either for most of the time (quasi-relaxed)
    relaxed = False
    relax_iters = 0
    while not relaxed:
        magnetisation, energy, lattice = metropolis(lattice, kernel, n_runs, bj, initial_energy)
        # recompute the total energy
        initial_energy = get_energy(lattice, kernel)
        mean_mag = np.mean(np.abs((magnetisation / n ** 2)[relaxation_range:]))
        if np.any([
            mean_mag > 1 - relaxation_tol,
            mean_mag < relaxation_tol,
            relax_iters > max_relaxation_iters,
            disable_relaxation
        ]):
            relaxed = True
            if relax_iters > max_relaxation_iters:
                print(f"Warning: System is potentially not relaxed. T = {1/bj:.3f}.")
        relax_iters += 1

    relaxed = (relax_iters - 1) > max_relaxation_iters
    tau = get_autocorr_time(magnetisation)

    ### MEASURE
    block_length = 16 * tau
    n_blocks = 100
    n_runs = int(block_length * n_blocks)

    magnetisation, energy, final_lattice = metropolis(lattice, kernel, n_runs, bj, initial_energy)

    # fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    # ax = axes[0]
    # ax.plot(savgol_filter(magnetisation / n ** 2, window_length=100, polyorder=1))
    # ax.set_xlabel('Algorithm Time Steps')
    # ax.set_ylabel(r'Average Spin $\bar{m}$')
    # ax.grid()
    # ax = axes[1]
    # ax.plot(savgol_filter(energy / n ** 2, window_length=100, polyorder=1))
    # ax.set_xlabel('Algorithm Time Steps')
    # ax.set_ylabel(r'Energy $E/J$')
    # ax.grid()
    # fig.tight_layout()
    # fig.suptitle(r'Evolution of Average Spin and Energy for $\beta J=$0.7', y=1.07, size=18)
    # plt.show()

    ### SAVE RUN
    save_ising(bj, n, bias, kernel_radius, kernel, n_dim, n_runs, block_length, relaxed, tau,
               magnetisation, energy, _id=_id)
    return


def save_ising(bj, n, bias, kernel_rad, kernel, n_dim, n_runs, block_length, relaxed, tau,
               magnetisation, energy,
               _id=0, file_dir="./runs/"):
    time = strftime('%d-%H-%M-%S', gmtime())
    name = f"T={1 / bj:.2f}_n={n}_bias={bias:.2f}_kernel-rad={kernel_rad:.1f}_ndim={n_dim}".replace(".", ",")
    name = name + f"_time={time}_id={_id}.h5"
    file_location = Path(file_dir + name)
    file = h5py.File(file_location, "w")

    ### save simulation data to h5 file
    meta_dict = {"T": 1 / bj,
                 "BJ": bj,
                 "n": n,
                 "bias": bias,
                 "kernel_rad": n_dim,
                 "kernel_shape": kernel.shape,
                 "kernel": kernel.flatten(),
                 "n_runs": n_runs,
                 "block_length": block_length,
                 "relaxed": relaxed,
                 "ext_field": False,
                 "auto_corr_time": tau,
                 "time": strftime('%Y-%m-%d-%H-%M-%S', gmtime()),
                 "id": _id
                 }

    # Store metadata in hdf5 file
    for k in meta_dict.keys():
        file.attrs[k] = meta_dict[k]

    for data, name in zip((magnetisation, energy),
                          ("magnetisation", "energy")):
        file.create_dataset(name,
                            data=data,
                            dtype="float32",
                            compression="gzip",
                            compression_opts=3)


if __name__ == "__main__":
    side_length = 50
    number_dimensions = 2
    run_ising(side_length, number_dimensions, bj=1./1.)
