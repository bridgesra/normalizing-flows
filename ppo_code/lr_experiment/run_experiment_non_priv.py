import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from timeit import default_timer as timer
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #supress future warnings
warnings.simplefilter(action='ignore', category=UserWarning) #supress user warnings


# %% 
# add our ./code folder to sys.path so we can import our modules: 
sys.path.append(Path(".", "code").absolute().as_posix())
sys.path.append(Path(".", "code", "lr_experiment").absolute().as_posix())
# sys.path.append(Path("..", "code").absolute().as_posix())
# sys.path.append(Path("..", "code", "lr_experiment").absolute().as_posix())

from data_utils import * # our own data utilities
from config import * #experiment configuration variables & results path
from regression_models import *
from experiment_utils import * # this has functions to load the data, utility functions (util_n) to be optimized, their sensitivities (sn)

# it also defines the potential function (log of expM probability numerator)
# it defines training loop function, and kernel function for running the experiment with multiprocessing


if __name__ == '__main__':
    
    jsonify({"regularization": "l1", 
             "normalization" :"l-infinty norm",
             "uses weights" : "true",
             "util": "l2",
             'epochs' : epochs, 
            'learning_rate': learning_rate, # = .01 #learning rate 
            'momentum': momentum, # = .001 #momentum
            'c': c, # = 0.1 # regularization constant
            'batch_size': batch_size #  = 100
             }, Path(non_priv_results_path, "experiment_metadata.json"))  # 

    print(f"Beginning Baseline experiment ...")
    
    metadata_path = Path(non_priv_results_path, "losses") # put model and losses data in this subfolder.
    if not metadata_path.is_dir():metadata_path.mkdir()

    save_path = Path(non_priv_results_path, "raw_results.json").as_posix()

    #set up for kernel
    #number of different models we will train
    args_list = [(model_index, metadata_path) for model_index in range(0,1000)]

    processes = cpu_count() - 2 # use all but one of my computer's cores
    pool = Pool(processes=processes)

    start = timer()
    print(f'\t\tstarting to train models, parallelized with {processes} cores ...')
    list_of_dicts = pool.starmap(non_priv_kernel_fn, args_list)
    end = timer()
    print(f"took {end-start}s start to finish")

    print(f'Saving results to {save_path}')

    jsonify(list_of_dicts, save_path)
