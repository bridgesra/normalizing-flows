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
             "util_0": "negative loss",
             "delta" : float(delta),
             "util_1": "negative sqrt loss",
             }, Path(dpsgd_hyperparameter_results_path, "experiment_metadata.json"))  # 

    # because the utilities should be the same here we only explore
    # hyperparameters for the first utility.
    for util_index, util in enumerate([L_from_pred]):
        print(f"Beginning util {util_index} ...")

                # makes path to store model and losses info:
        util_path = Path(dpsgd_hyperparameter_results_path, f'util_{util_index}_model')  # put raw results here
        if not util_path.is_dir():util_path.mkdir()
    
        metadata_path = Path(util_path, "losses") # put model and losses data in this subfolder.
        if not metadata_path.is_dir():metadata_path.mkdir()

        #set up for kernel
        #number of different models we will train
        # for hyperparameter search we just look at 10 per possible parameters
        #args_list = [(model_index, util_index, util, metadata_path) for model_index in range(0,1000)]
        args_list = []
        # grid search hyperparameters for rmsprop
        # because we keep the betas every epoch, we don't need to search over the epochs
        epochs = 50 # the maximum amount we'd want to train for
        param_index = 0
        for batch_size in [50, 100, 250, 500]: 
            for learning_rate in [0.0001, 0.001, 0.01, 0.1]:
                for momentum in np.arange(.01, .1, .025):
                    for alpha in [.01, .25, .5, .75, .99]:
                            param_index += 1
                            for model_index in range(0, 13):
                                param_dict = dict(lr = learning_rate,
                                                  momentum = momentum,
                                                  alpha = alpha)
                                args_list.append((model_index, util_index, util, param_index, param_dict, metadata_path, epochs, batch_size))


        processes = cpu_count() - 2 # use all but one of my computer's cores
        pool = Pool(processes=processes)
        
        start = timer()
        print(f'\t\tstarting to train models, parallelized with {processes} cores ...')
        list_of_dicts = pool.starmap(dpsgd_hyperparameter_search_kernel_fn, args_list)
        end = timer()
        print(f"took {end-start}s start to finish")

        save_path = Path(util_path, "raw_results.json").as_posix()
        print(f'Saving results to {save_path}')
        
        jsonify(list_of_dicts, save_path)



                                                  


    





