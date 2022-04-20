import numpy as np
import h5py
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from pathlib import Path

def read_h5_data(loc, ):
    """
    Should read data that is like:
        (data, steps)
    :param loc:
    :return: data like: (data, steps)
    """

    with h5py.File(loc, "r") as file:
        magnetization = np.array(file["magnetisation"])
        energy = np.array(file["energy"])

        header = dict(file.attrs)

    return header, magnetization, energy


def sort_results(headers, mags, ens):
    """
    currently not in use
    Order results depending on their temperature to find confidence
    :param headers:
    :param mags:
    :param ens:
    :return:
    """
    # # TODO: taken from
    # #  https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
    # # creates an array of indices, sorted by unique element
    # idx_sort = np.argsort(temperatures)
    # # sorts records array so all unique elements are together
    # sorted_records_array = temperatures[idx_sort]
    # # returns the unique values, the index of the first occurrence of a value, and the count for each element
    # vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)
    # # splits the indices into separate arrays
    # idxs = np.split(idx_sort, idx_start[1:])
    # print(idxs)
    return headers, mags, ens

def plot_tau(headers):
    temperatures = np.zeros_like(headers)
    taus = np.zeros_like(headers)
    for i, head in enumerate(headers):
        temperatures[i] = head["T"]
        taus[i] = head["auto_corr_time"]

    m = np.unique(temperatures).size
    temperatures, taus = temperatures.reshape((m, -1)).astype(float), taus.reshape((m, -1)).astype(float)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4),
                           constrained_layout=True)

    mean_tau = np.mean(taus, axis=1)
    std_tau = np.std(taus, axis=1)
    _temperatures = temperatures[:, 1]

    ax.plot(_temperatures, mean_tau,
            marker="o", fillstyle="full", markerfacecolor='white', markersize=5, color='black',
            label=r"T-$\tau$ plot ($J=1,~r_{k}=1$)")

    n_sigma = 1.
    ax.fill_between(_temperatures,
                    mean_tau + n_sigma * std_tau,
                    np.clip(mean_tau - n_sigma * std_tau, a_min=0., a_max=None),
                    color="gray", alpha=0.1,
                    label=f"{int(n_sigma)} "r"$\sigma$-CI")
    ax.axvline(2.269, c="black", ls="dotted",
               label=r"$T_{c}$")

    ax.set_xlabel('T [-]')
    ax.set_ylabel(r'$\tau$ [-]')

    ax.minorticks_on()
    ax.ticklabel_format(axis="both",
                        style="sci", scilimits=(-1, 3),
                        useMathText=True, useOffset=False)
    ax.legend()

    # plt.savefig('T-tau_plot.png', transparent=False, dpi=200, bbox_inches="tight")
    plt.show()


def plot_mean_abs_spin(headers, magnetizations):
    temperatures = np.zeros_like(headers)
    ns = np.zeros_like(headers)
    for i, head in enumerate(headers):
        temperatures[i] = head["T"]
        ns[i] = np.square(head["n"])

    mags = np.array([np.mean(np.abs(mag)) for mag in magnetizations]) / ns

    m = np.unique(temperatures).size
    temperatures, mags = temperatures.reshape((m, -1)).astype(float), mags.reshape((m, -1)).astype(float)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4),
                           constrained_layout=True)

    mean_mag = np.mean(mags, axis=1)
    std_mag = np.std(mags, axis=1)
    _temperatures = temperatures[:, 1]

    ax.plot(_temperatures, mean_mag,
            marker="o", fillstyle="full", markerfacecolor='white', markersize=5, color='black',
            label=r"T-$\langle|m|\rangle$ plot ($J=1,~r_{k}=1$)")

    n_sigma = 10.
    ax.fill_between(_temperatures,
                    mean_mag + n_sigma * std_mag,
                    np.clip(mean_mag - n_sigma * std_mag, a_min=0., a_max=None),
                    color="gray", alpha=0.1,
                    label=f"{int(n_sigma)} "r"$\sigma$-CI")
    ax.axvline(2.269, c="black", ls="dotted",
               label=r"$T_{c}$")

    ax.set_xlabel('T [-]')
    ax.set_ylabel(r'$\langle|m|\rangle$ [-]')

    ax.minorticks_on()
    ax.ticklabel_format(axis="both",
                        style="sci", scilimits=(-1, 3),
                        useMathText=True, useOffset=False)
    ax.legend()

    # plt.savefig('T-mag_plot.png', transparent=False, dpi=200, bbox_inches="tight")
    plt.show()


def plot_energy_spin(headers, energies):
    temperatures = np.zeros_like(headers)
    ns = np.zeros_like(headers)
    for i, head in enumerate(headers):
        temperatures[i] = head["T"]
        ns[i] = np.square(head["n"])

    ens = np.array([np.mean(en) for en in energies]) / ns

    fig, ax = plt.subplots(1, 1, figsize=(8, 4),
                           constrained_layout=True)

    from scipy.signal import savgol_filter
    for energy in ens:
        plt.axhline(energy)
        # try:
        #     plt.axhline(np.mean(savgol_filter(energy, window_length=int(energy.size / 10) + 1, polyorder=1)))
        # except BaseException:
        #     plt.axhline(np.mean(savgol_filter(energy, window_length=int(energy.size / 10), polyorder=1)))

    plt.show()


    m = np.unique(temperatures).size
    temperatures, ens = temperatures.reshape((m, -1)).astype(float), ens.reshape((m, -1)).astype(float)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4),
                           constrained_layout=True)

    mean_ens = np.mean(ens, axis=1)
    std_ens = np.std(ens, axis=1)
    _temperatures = temperatures[:, 1]

    ax.plot(_temperatures, mean_ens,
            marker="o", fillstyle="full", markerfacecolor='white', markersize=5, color='black',
            label=r"T-$e$ plot ($J=1,~r_{k}=1$)")

    n_sigma = 3.
    ax.fill_between(_temperatures,
                    mean_ens + n_sigma * std_ens,
                    mean_ens - n_sigma * std_ens,
                    color="gray", alpha=0.1,
                    label=f"{int(n_sigma)} "r"$\sigma$-CI")
    ax.axvline(2.269, c="black", ls="dotted",
               label=r"$T_{c}$")

    ax.set_xlabel('T [-]')
    ax.set_ylabel(r'$e$ [-]')

    ax.minorticks_on()
    ax.ticklabel_format(axis="both",
                        style="sci", scilimits=(-2, 3),
                        useMathText=True, useOffset=False)
    ax.legend()

    # plt.savefig('T-en_plot.png', transparent=False, dpi=200, bbox_inches="tight")
    plt.show()


def get_results(results_dir="./runs/"):
    files = Path(results_dir).rglob(f"*.h5")
    files = np.array([path for path in files]).flatten()

    headers = []
    magnetizations = []
    energies = []

    for i, file in enumerate(files):
        header, magnetization, energy = read_h5_data(file)
        if "time=20-" in str(file):
            continue
        # appending these large arrays makes me unreasonably mad, but I am very tired
        headers.append(header)
        magnetizations.append(magnetization)
        energies.append(energy)

    plot_tau(headers)
    plot_mean_abs_spin(headers, magnetizations)
    plot_energy_spin(headers, energies)


if __name__ == "__main__":
    get_results()