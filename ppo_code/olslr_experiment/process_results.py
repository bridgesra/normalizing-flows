# %% 
from pathlib import Path
import pandas as pd 
import numpy as np 
## TODO some QAQC on the confidence intervals. count by hand, then see what we get in the plot
## See: https://www.kaggle.com/code/daryasikerina/data-visualization-with-seaborn
import seaborn as sns
import matplotlib.pyplot as plt
import sys

# add our ./code folder to sys.path so we can import our modules: 
sys.path.append(Path(".", "code").absolute().as_posix())
sys.path.append(Path(".", "code", "olslr_experiment").absolute().as_posix())
from data_utils import * # our own data utilities
from olslr_experiment.experiment_utils import *
from olslr_experiment.config import *

# %% 
def nf_process_results(results_path, util_index):
    util_path = Path(results_path, f'util_{util_index}_model')  # put raw results here
    # metadata_path = Path(util_path, "model_metadata") # put model and losses data in this subfolder.
    
    # read in results and make a df w/ error column
    raw = unjsonify(Path(util_path, 'raw_results.json'))
    records = [ {'util': d['util'], 'epsilon': d['epsilon'], 'beta': b}   for d in raw for b in d['betas']  ]
    df = pd.DataFrame.from_records(records)
    meta = unjsonify(Path(results_path, 'experiment_metadata.json'))
    df["error"] = df.apply(lambda row: 1-np.abs((row['beta'] - meta['beta_opt'])/meta['beta_opt']), axis = 1)
    return df


def dpsgd_process_results(dpsgd_results_path, noise_multiplier_index, eps_error = .01):
    """
    This function processes the results from the olslr dpsgd experiments. It reads the results, computes epsilon given the accountant history and returns a dataframe with epsilon values and the accuracy.

    Args:
       dpsgd_results_path (string): the path to the results that need processing
       noise_multiplier_index (int): which noise_multiplier experiment to read in
       eps_error (float): error for epsilon when computing use the PRV accounting method (default = .01)
    Returns:
       dataframe: all model results with error and epsilon 
    """

    dpsgd_noise_multiplier_path = Path(dpsgd_results_path, f'noise_multiplier_{noise_multiplier_index}')  # put raw results here

    #read in results and make a df w/ error column
    dpsgd_raw = unjsonify(Path(dpsgd_noise_multiplier_path, 'raw_results.json'))
    meta = unjsonify(Path(dpsgd_results_path, 'experiment_metadata.json'))
    
    # check the accountant history agrees for each model (it should)
    accountant_histories = [d['accountant_history'] for d in dpsgd_raw]
    a = accountant_histories[1]
    histories_all_equal = True
    for a_ in accountant_histories:
        if a_ != a:
            histories_all_equal = False
    if histories_all_equal:
        epsilons = [compute_epsilon(epoch, delta, noise_multiplier = a['noise_multiplier'], sample_rate = a['sample_rate'], steps_per_epoch = a['steps_per_epoch'], eps_error = eps_error) for epoch in range(dpsgd_epochs)]
        dpsgd_records = [{'noise_multiplier': a['noise_multiplier'],
                          'epsilon': epsilons[i],
                          'beta' : d['betas'][i]}
                         for d in dpsgd_raw for i in range(dpsgd_epochs)]
    else:
        # this would take a lot longer so we only do it if we have to
        print("Computing epsilons per model, per epoch because accountant histories differed. This may take a while...")
        dpsgd_records = [{'noise_multiplier' : a['noise_multiplier'],
                          'epsilon': compute_epsilon(i, delta,
                                                     noise_multiplier = a['noise_multiplier'],
                                                     sample_rate = a['sample_rate'],
                                                     steps_per_epoch = a['steps_per_epoch']),
                          'beta' : d['betas'][i]} for d in dpsgd_raw for i in range(dpsgd_epochs)]

    dpsgd_df = pd.DataFrame.from_records(dpsgd_records)
    dpsgd_df["error"] = dpsgd_df.apply(lambda row: 1-np.abs((row['beta'] - float(meta['beta_opt']))/float(meta['beta_opt'])), axis = 1)
    return dpsgd_df


def plot_raw(df, name, delta = None, noise_multiplier = None, label = None,
             color = 'blue', save_fig_path = None, fig_and_ax = None):
    """
    This function plots epsilon vs accuracy for the given dataframe.

    Args:
       df (dataframe): experiment results
       name (string): Name at top of title (usually DPSGD or NF)
       delta (float): delta used to compute epsilon (optional)
       noise_multiplier (float): noise multiplier used in DPSGD (optional)
       label (string): Legend label (default = None)
       color (string): Plot color (default = 'blue')
       save_fig_path (string): where to save the plot (optional)
       fig_and_ax (tuple): Should be the result of plt.subplots(), allows combining plots
    Returns:
       (fig, ax): the resulting plot
    """

    if fig_and_ax == None:fig_and_ax = plt.subplots()
    fig, ax = fig_and_ax
    sns.regplot(ax = ax, x = "epsilon", y = "error", data = df, marker = "o", label = label,
                fit_reg=False,  scatter_kws={'alpha':.5, 'linewidths': 1, 'color':color}, ci = 99)
    ax.set_title(f"{name}:Raw Data + Regression\nAccuracy (|beta-beta_optimal|) vs. Privacy (epsilon)\n")
    ax.legend()
    if delta != None:
        if noise_multiplier != None:
            caption = f"delta = {delta}, noise multiplier = {noise_multiplier}"
        else:
            caption = f"delta = {delta}"
        fig.text(.5, -.02, caption, ha='center')

    if save_fig_path != None:
        plt.figure(constrained_layout=True)
        fig.savefig(save_fig_path)

    return (fig, ax)


def plot_99_conf(df, noise_multiplier = None, delta = None,
                 label = None, color = 'blue', save_fig_path = None, fig_and_ax = None):
    """
    This function plots 99% conf interval for epsilon vs accuracy for the given dataframe.

    Args:
       df (dataframe): experiment results
       name (string): Name at top of title (usually DPSGD or NF)
       delta (float): delta used to compute epsilon (optional)
       noise_multiplier (float): noise multiplier used in DPSGD (optional)
       label (string): Legend label (default = None)
       color (string): Plot color (default = 'blue')
       save_fig_path (string): where to save the plot (optional)
       fig_and_ax (tuple): Should be the result of plt.subplots(), allows combining plots
    Returns:
       (fig, ax): the resulting plot
    """
    if fig_and_ax == None:fig_and_ax = plt.subplots(layout = "constrained")
    fig, ax = fig_and_ax
    sns.regplot(ax = ax, x = "epsilon", y = "error", data = df,
                marker = "_", fit_reg=False,
                scatter_kws={'alpha':.9, 'linewidths': .5, 'color':color},
                x_ci = 99, ci = 1,
                x_bins = 40, label = label)
    ax.set_title(f"99% Confidence Intervals\nRelative Difference vs. Privacy \n(1-|(beta-beta_optimal)/beta_optimal|) vs (epsilon)")
    ax.legend()
    if delta != None:
        if noise_multiplier != None:
            caption = f"delta = {delta}, noise multiplier = {noise_multiplier}"
        else:
            caption = f"delta = {delta}"
        fig.text(.5, -.025, caption, ha='center')
    if save_fig_path != None:
        fig.savefig(save_fig_path, bbox_inches='tight')
    return fig, ax


def plot_nf_and_dpsgd(nf_df, dpsgd_df, thres = .05, save_fig_path = None, plot_fn = plot_99_conf):
    """
    This function plots epsilon vs accuracy for both NF and DPSGD experiments. It will ensure that it only plots epsilons that are (nearly) found in both experiments as defined by thres.

    Args:
       nf_df (dataframe): normalizing flow experiment results
       dpsgd_df (dataframe): DPSGD experiment results
       thres (float): how close epsilons from DPSGD experiment should be to NF epsilons (default = .05)
       save_fig_path (string): where to save the plot (optional)
       plot_fn (function): what plotting function to use (see plot_99_conf or plot_raw)
    Returns:
       (fig, ax): the resulting plot
    """


    # filter dpsgd and nf dataframes to have similar epsilons
    nf_eps = nf_df.epsilon.copy()
    dpsgd_eps = dpsgd_df.epsilon.copy()
    nf_eps = nf_eps.unique()
    nf_eps.sort()
    dpsgd_eps = dpsgd_eps.unique()
    dpsgd_eps.sort()
    # only keeps epsilons that are close together in both experiments
    # this is necessary because DPSGD epsilons won't follow a nice spacing like
    # NF epsilons will
    to_keep_nf_eps = []
    to_keep_dpsgd_eps = []

    state = (False, 0, 0, 0, math.inf) #(found, prev_ind, curr_ind, prev_closeness, curr_closeness)
    for eps in dpsgd_eps:
        state = (False, state[1], state[1], state[3], math.inf)
        while not state[0] and state[2] <= len(nf_eps)-1:
            d = abs(nf_eps[state[2]] - eps)
            if d <= thres:
                if state[1]== state[2]:
                    # did we find a closer epsilon?
                    if d < state[3]:
                        to_keep_dpsgd_eps.pop() # remove previous epsilon
                        to_keep_dpsgd_eps.append(eps) # push closer epsilon
                else:
                    to_keep_dpsgd_eps.append(eps)
                    to_keep_nf_eps.append(nf_eps[state[2]])
                state  = (True, state[2], state[2], d, math.inf)
            else:
                if d > state[4]: # we are getting farther away so we won't ever find something
                    state = (False, state[1], len(nf_eps), 0, math.inf)
                else:
                    state = (False, state[1], state[2]+1, 0, d)

    
    to_plot_nf = nf_df[nf_df.epsilon.isin(to_keep_nf_eps)]
    to_plot_dpsgd = dpsgd_df[dpsgd_df.epsilon.isin(to_keep_dpsgd_eps)]
    noise_multiplier = dpsgd_df['noise_multiplier'][1] # they are all the same value

    fig, ax = plt.subplots()
    plot_fn(to_plot_nf, "NF", label = 'Exp+NF', color = 'blue', fig_and_ax = (fig, ax))
    plot_fn(to_plot_dpsgd, "DPSGD", label = 'DPSGD', 
            color = 'green', fig_and_ax = (fig, ax))
    ax.set_title("99% Confidence Intervals\nAccuracy (|beta-beta_optimal|) vs. Privacy (epsilon)")
    fig.text(.5, -.02, f"DPSGD: delta = {delta}, noise multiplier = {noise_multiplier}", ha='center')
    ax.legend()
    #fig.show()
    if save_fig_path != None:
        plt.figure(constrained_layout=True)
        fig.savefig(save_fig_path, bbox_inches="tight")
    return fig, ax



if __name__ == '__main__':

     nf_df = nf_process_results(results_path, 0)
     dpsgd_df0 = dpsgd_process_results(dpsgd_results_path, 0)
     dpsgd_df1 = dpsgd_process_results(dpsgd_results_path, 1)
     dpsgd_df2 = dpsgd_process_results(dpsgd_results_path, 2)

     plot_99_path = Path(plots_path, "99_conf_plots")
     if not plot_99_path.is_dir():plot_99_path.mkdir()
     plot_99_conf(nf_df[nf_df.epsilon <= .2], label = 'ExpM+NF',
                  save_fig_path = Path(plot_99_path, "exp_nf_99_conf_plot_eps_to_dot02_rel.png"))

     plot_99_conf(nf_df[(nf_df.epsilon >= .2) & (nf_df.epsilon <=2) ], label = 'ExpM+NF',
                  save_fig_path = Path(plot_99_path, "exp_nf_99_conf_plot_eps_in_dot2_to_2_rel.png"))

     plot_99_conf(nf_df[(nf_df.epsilon >= 3) & (nf_df.epsilon <=10) ], label = 'ExpM+NF',
                  save_fig_path = Path(plot_99_path, "exp_nf_99_conf_plot_eps_in_3_to_10_rel.png"))

     plot_99_conf(dpsgd_df0, label = "DPSGD", delta = delta, color = 'green',
                  noise_multiplier = dpsgd_df0['noise_multiplier'][1],
                  save_fig_path = Path(plot_99_path, "dpsgd_99_conf_plot_eps_in_dot2_to_2_rel.png"))
    
     plot_99_conf(dpsgd_df1, label = "DPSGD", delta = delta, color = 'green',
                  noise_multiplier = dpsgd_df1['noise_multiplier'][1],
                  save_fig_path = Path(plot_99_path, "dpsgd_99_conf_plot_eps_in_3_to_11_rel.png"))

     plot_99_conf(dpsgd_df2, label = "DPSGD", delta = delta, color = 'green',
                  noise_multiplier = dpsgd_df2['noise_multiplier'][1],
                  save_fig_path = Path(plot_99_path, "dpsgd_99_conf_plot_eps_to_dot1_rel.png"))

     plot_nf_and_dpsgd(nf_df, dpsgd_df0, save_fig_path = Path(plot_99_path, "exp_nf_dpsgd_99_conf_plot_eps_in_dot2_to_2_rel.png"))

     plot_nf_and_dpsgd(nf_df, dpsgd_df1, thres = .1, save_fig_path = Path(plot_99_path, "exp_nf_dpsgd_99_conf_plot_eps_in_3_to_10_rel.png"))

     plot_nf_and_dpsgd(nf_df, dpsgd_df2, thres = .002, save_fig_path = Path(plot_99_path, "exp_nf_dpsgd_99_conf_plot_eps_to_dot1_rel.png"))










# %%
