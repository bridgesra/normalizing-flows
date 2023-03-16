"""
Runs Oridinary Least Squares Linear Regression Experiment with ExpM. 
See ./code/olslr_experiment/config.py for experiment configurations. 
Writes results to ./results/olslr_expm_experiment
"""

import torch
import sys
from pathlib import Path
from multiprocessing import Pool, cpu_count
from timeit import default_timer as timer

# add our ./code folder to sys.path so we can import our modules:
sys.path.append(Path(".", "code").absolute().as_posix())
from config import * #experiment configuration variables & results path
from normalizing_flows import * # classes for building NF models
from experiment_utils import * # this loads the data, utility functions (util_n) to be optimized, their sensitivities (sn)
# it also defines the potential function (log of expM probability numerator)
# it defines training loop function, and kernel function for running the experiment with multiprocessing


if __name__ == '__main__': 
	# ground truth beta:
	b_opt = torch.dot(torch.mm(torch.linalg.inv(torch.mm(X.T, X)), X.T).squeeze(0), y)
	print(f'Running OLSLR ExpM Experiment ... /nOptimal beta is {b_opt}')

	# store experiment info:
	jsonify({"beta_opt": float(b_opt),
		"util_0": "negative loss",
		"s0": s0,
		"util_1": "negative sqrt loss",
		"s1": s1}, Path(results_path, "experiment_metadata.json"))

	for util_index, (util, s) in enumerate([(util_0, s0), (util_1, s1)]):
	# util_index, util, s = 0, util_0, s0

		print(f"Beginning util {util_index} ...")

		# makes path to store model and losses info:
		util_path = Path(results_path, f'util_{util_index}_model')  # put raw results here
		if not util_path.is_dir():util_path.mkdir()

		metadata_path = Path(util_path, "model_metadata") # put model and losses data in this subfolder.
		if not metadata_path.is_dir(): metadata_path.mkdir()

		# set up args for kernel: 
		args_list = [(util_index, util, s, epsilon,  metadata_path) for epsilon in epsilons] #epsilons defined in config.  

		# set up multiprocessing
		processes = cpu_count() - 1  # use all but one of my computer's cores
		pool = Pool(processes=processes)

		# run the multiprocesing: 
		start = timer()
		print(f'\t\tstarting to train NF, then sample from NFs, parallelized with {processes} cores ...')
		# should return a list of the output dicts from each pair of vendors
		list_of_dicts = pool.starmap(kernel_fn, args_list)
		end = timer()
		print(f"took {end-start}s start to finish")

		# save results: 
		save_path = Path(util_path, "raw_results.json").as_posix()
		print(f'Saving results to {save_path}')
		jsonify(list_of_dicts, save_path)
