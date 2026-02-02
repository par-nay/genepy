import numpy as np
import ga_utils as poppy 

def trial_fn(x):
    if type(x) is np.ndarray:
        N_var = x.shape[-1]