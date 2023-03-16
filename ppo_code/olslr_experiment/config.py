# config for OLSLR Experiment 
from pathlib import Path
import numpy as np 

# epsilon range for NF model: 
epsilons = list(np.arange(.02, .2, .02))+list(np.arange(.2, 10.2, .2))
		
# hyperparameters for NF model: 
dim = 1
n_flows = 10 

## TODO change the NF source code to accept a ModuleList
# hyperparameters for NF training (# from Rezende et al.)
learning_rate = 1e-5 
momentum = .9 
epochs = 20000 # will have 1 batch per epoch 
batch_size = 100 # number of samples to use per batch = per epoch


results_path = Path(".", "results", "olslr", "olslr_expm_experiment")  # Results folder path
if not results_path.is_dir():
	results_path.mkdir()

#### DPSGD experiment hyperparameters:

# hyperparameters for DPSGD training (see hyperparameter section in notebook
# for explanation on how there were choosen)
dpsgd_learning_rate = .0001 
dpsgd_momentum = 0.060000000000000005
dpsgd_alpha = .5
dpsgd_epochs = 75 # will have all the batches per epoch
dpsgd_batch_size = 50 # number of samples to use per batch
dpsgd_noise_multiplier0 = 1 # will get epsilon to ~2 in 75 epochs
dpsgd_noise_multiplier1 = .553 # will get epsilon to ~10 in 75 epochs
dpsgd_noise_multiplier2 = 12 # will get epsilon to ~.1 in 75 epochs

delta = 1e-5 # usually recommended to be no larger than 1/N 

dpsgd_results_path = Path(".", "results", "olslr", "olslr_dpsgd_experiment")  # Results folder path
if not dpsgd_results_path.is_dir():
	dpsgd_results_path.mkdir()

dpsgd_hyperparameter_results_path = Path(".", "results", "olslr", "olslr_hyparameter_search_dpsgd")
if not dpsgd_hyperparameter_results_path.is_dir():
    dpsgd_hyperparameter_results_path.mkdir()

plots_path = Path(".", "results", "olslr", "olslr_plots")
if not plots_path.is_dir():plots_path.mkdir()
