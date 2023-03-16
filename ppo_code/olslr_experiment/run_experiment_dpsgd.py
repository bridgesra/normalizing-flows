import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import pandas as pd
from multiprocessing import Pool, cpu_count
from timeit import default_timer as timer



# %% 
# add our ./code folder to sys.path so we can import our modules: 
sys.path.append(Path(".", "code").absolute().as_posix())
from data_utils import * # our own data utilities
from config import * #experiment configuration variables & results path
from linear_regression_model import *
from experiment_utils import * # this loads the data, utility functions (util_n) to be optimized, their sensitivities (sn)

# it also defines the potential function (log of expM probability numerator)
# it defines training loop function, and kernel function for running the experiment with multiprocessing

data = DatasetMaker(X, y)
dataloader = DataLoader(dataset = data, batch_size = batch_size, shuffle = True)

if __name__ == '__main__':
    # ground truth beta:
    b_opt = torch.dot(torch.mm(torch.linalg.inv(torch.mm(X.T, X)), X.T).squeeze(0), y)
    print(f'Optimal beta is {b_opt}')
    jsonify({"beta_opt": float(b_opt),
             "loss": "negative loss",
             "delta" : float(delta),
             "learning_rate": float(dpsgd_learning_rate),
             "momentum": float(dpsgd_momentum),
             "alpha":float(dpsgd_alpha),
             "epochs":float(dpsgd_epochs),
             "batchsize":float(dpsgd_batch_size)
             }, Path(dpsgd_results_path, "experiment_metadata.json"))  # 


    for noise_multiplier_index, noise_multiplier in enumerate([dpsgd_noise_multiplier0, dpsgd_noise_multiplier1, dpsgd_noise_multiplier2]): 
        print(f"Beginning noise multiplier = {noise_multiplier} ...")

        # makes path to store model and losses info:
        noise_multiplier_path = Path(dpsgd_results_path, f'noise_multiplier_{noise_multiplier_index}')  # put raw results here
        if not noise_multiplier_path.is_dir():noise_multiplier_path.mkdir()
    
        metadata_path = Path(noise_multiplier_path, "losses") # put model and losses data in this subfolder.
        if not metadata_path.is_dir():metadata_path.mkdir()

        save_path = Path(noise_multiplier_path, "raw_results.json").as_posix()

        #set up for kernel
        #number of different models we will train
        args_list = [(model_index, noise_multiplier, metadata_path) for model_index in range(0,1000)]

        processes = cpu_count() - 1 # use all but one of my computer's cores
        pool = Pool(processes=processes)
        
        start = timer()
        print(f'\t\tstarting to train models, parallelized with {processes} cores ...')
        list_of_dicts = pool.starmap(dpsgd_kernel_fn, args_list)
        end = timer()
        print(f"took {end-start}s start to finish")

        print(f'Saving results to {save_path}')
        jsonify(list_of_dicts, save_path)



                                                  


    





