import numpy as np
import os 
from joblib_progress import joblib_progress
from sklearn.model_selection import ParameterGrid
from joblib import Parallel, delayed


def fit(params: dict):
    instruction = "python train.py"
    for key, val in params.items():
        instruction += f" --{key.replace('_', '-')} {val}" 
    instruction += " --epochs 100"
    instruction += " --checkpoint"
    instruction += " --send-mail"
    instruction += " --method td3"
    instruction += " --seed 123"
    os.system(instruction)


if __name__ == "__main__":
    param_grid = dict(
        polyak=np.array([1e-2, 1e-3, 1e-4]),
        policy_delay=np.array([50, 100, 1000], dtype=int),
        pi_lr=np.array([1e-3, 1e-4, 1e-5, 1e-6]),
        q_lr=np.array([1e-2, 1e-3]),
        update_every=np.array([1000, 2000]),
        act_noise=np.array([0.01, 0.05, 0.1]),
        target_noise=np.array([0.01, 0.05, 0.1, 0.2]),
        noise_clip=np.array([0.1, 0.3])
    )

    param_candidates = ParameterGrid(param_grid)

    print(f'{len(param_candidates)} candidates')
    with joblib_progress("Searching parameters...", total=len(param_candidates)):
        Parallel(n_jobs=4, verbose=10)(delayed(fit)(params) for params in param_candidates)

