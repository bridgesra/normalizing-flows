"""
Runs the Exp+NF Logistic Regression Experiment on the Adult Dataset
It minimizes Logistic L2 loss with L1 regularization. 
It uses ReverseKL to train the NF. 


See .code/lr_experiment/config.py for experiment configurations 

Results stored in config.expm_results_path = ./results/logistic-regression-experiment

"""

import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from timeit import default_timer as timer

# add our ./code folder to sys.path so we can import our modules:
sys.path.append(Path(".", "code").absolute().as_posix())
from normalizing_flows import * # classes for building NF models

# add our ./code/lr_experiment/ folder 
sys.path.append(Path(".", "code", "lr_experiment").absolute().as_posix())
from config import * #experiment configuration variables & results path
from experiment_utils import * # this loads utility function (util), potential function (pot) to be optimized, 
# it defines training loop function, and kernel function for running the experiment with multiprocessing

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #supress future warnings


if __name__ == '__main__': 
	meta = {"regularization": "l1", # on beta 
			"normalization" :"l-infinty norm", # on X
			"uses weights" : "true", # data ships with weights
			"util": "l2 loss", 
			's': s,  # sensitivity for L2 loss.
			'epsilons': epsilons, 
			'n_flows': n_flows, 
			'nf_learning_rate': nf_learning_rate, 
			'nf_momentum' : nf_momentum,
			"nf_epochs":nf_epochs, 
			"nf_batch_size":nf_batch_size, 
			"nf_c": nf_c
		}	
	print(f"Beginning ExpM + NF experiment with \n{meta} ")

	#save experiment metadata
	jsonify(meta, Path(expm_results_path, "experiment_metadata.json"))  # 

	# read in data, this makes a stratified train/test split 80/20% (both have 25% 1s). 
	df, training_data, test_data = load_data() # function in experiment_utils.py

	# Make Pytorch Dataloader (batches) 
	train_dataloader = DataLoader(training_data, batch_size = len(training_data) ) # one batch--used for evaluating the NF loss func.
	test_dataloader = DataLoader(test_data, batch_size=len(test_data)) # one batch for test data 

	# make the training set tensors: 
	X_train_w, y_train = next(iter(train_dataloader))
	X_train = X_train_w[:,:-1] # feature vectors
	w_train = X_train_w[:,-1] # weights
	print(X_train.shape, y_train.shape, w_train.shape)
	
	dim = X_train.shape[1] + 1
 	
  	# set up args for kernel: 
	args_list = [ ( epsilon, X_train, y_train, w_train) for epsilon in epsilons ]
	# set up multiprocessing
	processes = cpu_count() - 2  # use all but two of my computer's cores
	pool = Pool(processes=processes)

	#instantiate model
	model = NormalizingFlow(dim, n_flows=n_flows) # number of layers/flows = n_flows

	# run the multiprocesing: 
	start = timer()
	print(f'\t\tstarting to train NF, then sample from NFs, parallelized with {processes} cores ...')
	# should return a list of the outputs 
	list_of_dicts = pool.starmap(kernel_fn, args_list)
	end = timer()
	print(f"took {end-start}s start to finish")

	# save results: 
	save_path = Path(expm_results_path, "raw_results.json").as_posix()
	print(f'Saving results to {save_path}')
	jsonify(list_of_dicts, save_path)
 