# config file for LR experiment
from pathlib import Path
import numpy as np

# raw_data paths: 
ADULT_DATA_PATH = Path(".","data", "adult_dataset", "raw", "adult.data")
ADULT_TEST_PATH = Path(".","data", "adult_dataset", "raw", "adult.test")

# column names: 
COLS = ['age', 'workclass', 'fnlwgt', "education", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "captial_gain", "capital_loss", "hours_per_week", "native_country", "target"]

# cleaned data path
ADULT_CLEAN_FULL_PATH = Path(".","data", "adult_dataset", 'adult_clean_full.csv')
if not ADULT_CLEAN_FULL_PATH.is_file():
    ADULT_CLEAN_FULL_PATH = Path("..", "..","data", "adult_dataset", 'adult_clean_full.csv')
    

# Baseline no privacy hyperparameters
epochs = 51 # epochs
learning_rate = .01 #learning rate 
momentum = .001 #momentum
c = 0.1 # regularization constant
batch_size = 1000

non_priv_results_path = Path(".", "results", "lr", "lr_non_priv_experiment")  # Results folder path
if not non_priv_results_path.is_dir():
    non_priv_results_path.mkdir()

##### DPSGD globals
#TODO grid search to make sure these are good choices
# actually, Bobby did grid search with scikit learn and we beat that
# and our private model performs as good or better than a non-private
# pytorch model so we just use these.
dpsgd_epochs = 26 # epochs
dpsgd_learning_rate = .01 #learning rate 
dpsgd_momentum = .001 #momentum
dpsgd_c = 0.1 # regularization constant
delta = 1e-5 # less than 1/N
dpsgd_batch_size = 100
dpsgd_noise_multiplier0 = .852 # will get epsilon to ~2 in 26 epochs
dpsgd_noise_multiplier1 = 1.5 # will get epsilon to ~.75 in 26 epochs
dpsgd_noise_multiplier2 = 9.25 # will get epsilon to ~.1 in 26 epochs
dpsgd_noise_multiplier3 = 1 # will get epsilon to ~1.4 in 26 epochs

dpsgd_results_path = Path(".", "results", "lr", "lr_dpsgd_experiment")  # Results folder path
if not dpsgd_results_path.is_dir():
    dpsgd_results_path.mkdir()


# ExpM+NF globals
s = 2 # sensitivity for L2 loss. 
epsilons = list(np.arange(.2, 2.1, .2))
# hyperparameters for NF model: 
n_flows = 10
# hyperparameters for NF training (# from Rezende et al.)
nf_learning_rate = 1e-4
nf_momentum = .6
nf_epochs = 25000 # will have 1 batch per epoch
nf_batch_size = 500 # number of samples to use per batch = per epoch
nf_c = .1

expm_results_path = Path(".", "results", "lr", "lr_expm_experiment_1")  # Results folder path
if not expm_results_path.is_dir():
	expm_results_path.mkdir()
expm_metadata_path = Path(expm_results_path, 'nf_training_metadata')
if not expm_metadata_path.is_dir():
	expm_metadata_path.mkdir()

plots_path = Path(".", "results", "lr", "lr_plots")
if not plots_path.is_dir():plots_path.mkdir()
