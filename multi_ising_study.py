from ising_model import *


def run_many_isings():
    step = 0.1
    temp = np.arange(1., 4. + step, step)
    repeats = 10
    print('Temperature sweeping Ising Model...')
    i = 0

    for re in tqdm(np.arange(1, repeats+1)):
        for T in tqdm(temp, leave=False):
            run_ising(n=50, n_dim=2,
                      n_runs=int(1e4),
                      bj=1/T, bias=0.75, kernel_radius=1.,
                      _id=i)
            i += 1

    temp = [2. / np.log(1. + np.sqrt(2))]
    print('Run Ising Model at critical temperature ...')
    for re in tqdm(np.arange(1, repeats+1)):
        for T in tqdm(temp, leave=False):
            run_ising(n=50, n_dim=2,
                      n_runs=int(1e4),
                      bj=1/T, bias=0.75, kernel_radius=1.,
                      _id=i,
                      disable_relaxation=True)

            i += 1

if __name__ == "__main__":
    run_many_isings()