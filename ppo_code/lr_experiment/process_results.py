# process_results.py

# %%
import sys, torch, math
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
# add our ./code folder to sys.path so we can import our modules:
sys.path.append(Path(".", "code").absolute().as_posix())
from normalizing_flows import * # classes for building NF models

# add our ./code/lr_experiment/ folder 
sys.path.append(Path(".", "code", "lr_experiment").absolute().as_posix())
from config import * #experiment configuration variables & results path
from experiment_utils import * # this loads get_auc()

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning) #supress future warnings
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def get_nf_results(results_path):
    """
    This function processes the results from the lr exp+nf experiments. 
    It reads the results, computes auc and returns a dataframe with epsilon values and the auc.

    Args:
       results_path (string): the path to the results that need processing
    Returns:
       dataframe: all model results with auc and epsilon 
    """
    # read in data, this makes a stratified train/test split 80/20% (both have 25% 1s). 
    df, training_data, test_data = load_data() # function in experiment_utils.py

    # Make Pytorch Dataloader (batches) 
    test_dataloader = DataLoader(test_data, batch_size=len(test_data)) # one batch for test data 

    X_tw, y_test = next(iter(test_dataloader)) # has weights as last col of X
    X_test = X_tw[:,:-1] # feature vectors
    w_test = X_tw[:, -1] # weights
    print(X_test.shape, y_test.shape, w_test.shape)
    
    raw = unjsonify(Path(results_path, "raw_results.json").as_posix())
     
    records = [ {   
                'epsilon' : d['epsilon'], 
                'AUC': get_auc(torch.tensor(b), X_test, y_test, w_test) 
               } 
             for d in raw for b in d['betas']]
    df = pd.DataFrame.from_records(records)
    return df


def get_dpsgd_results(results_path, noise_multiplier_index, eps_error = .01):
    """
    This function processes the results from the lr dpsgd experiments. It reads the results, computes epsilon given the accountant history and returns a dataframe with epsilon values and the accuracy.

    Args:
       results_path (string): the path to the results that need processing
       eps_error (float): error for epsilon when computing use the PRV accounting method (default = .01)
    Returns:
       dataframe: all model results with auc and epsilon 
    """
    # read in data, this makes a stratified train/test split 80/20% (both have 25% 1s). 
    df, training_data, test_data = load_data() # function in experiment_utils.py

    # Make Pytorch Dataloader (batches) 
    test_dataloader = DataLoader(test_data, batch_size=len(test_data)) # one batch for test data 

    X_tw, y_test = next(iter(test_dataloader)) # has weights as last col of X
    X_test = X_tw[:,:-1] # feature vectors
    w_test = X_tw[:, -1] # weights
    
    nm_path = Path(results_path, f"noise_multiplier_{noise_multiplier_index}")
    raw = unjsonify(Path(nm_path, 'raw_results.json'))
    accountant_histories = [d['accountant_history'] for d in raw]
    a = accountant_histories[1]
    histories_all_equal = True

    meta = unjsonify(Path(results_path, f'experiment_metadata.json'))
    # check that histories agree... they should
    for a_ in accountant_histories:
        if a_ != a:
            histories_all_equal = False

    if histories_all_equal:
        epsilons = [compute_epsilon(epoch, delta,
                                    noise_multiplier = a['noise_multiplier'],
                                    eps_error = eps_error,
                                    sample_rate = a['sample_rate'],
                                    steps_per_epoch = a['steps_per_epoch'])
                    for epoch in range(meta['dpsgd_epochs'])]
        records = [{'AUC': get_auc(torch.tensor(d['betas'][i]), X_test, y_test, w_test),
                    'noise_multiplier' : a['noise_multiplier'],
                    'epsilon': epsilons[i]} for d in raw for i in range(meta['dpsgd_epochs'])]
    else:
        # this would take a lot longer so we only do it if we have to
        print("Computing epsilons per model, per epoch because accountant histories differed. This may take a while...")
        records = [{'AUC': get_auc(torch.tensor(d['betas'][i]), X_test, y_test, w_test),
                    'noise_multiplier' : d['accountant_history']['noise_multiplier'],
                    'epsilon': compute_epsilon(i, delta,
                                               noise_multiplier = d['accountant_history']['noise_multiplier'],
                                               eps_error = eps_error,
                                               sample_rate = d['accountant_history']['sample_rate'],
                                               steps_per_epoch = d['accountant_history']['steps_per_epoch'])}
                   for d in raw for i in range(meta['dpsgd_epochs'])]
    df = pd.DataFrame.from_records(records)
    return df

def get_non_priv_baseline_results(results_path):
    """
    This function processes the results from the lr non private experiments. 

    Args:
       results_path (string): the path to the results that need processing
    Returns:
       dataframe: all model results with auc and epsilon 
    """
    # read in data, this makes a stratified train/test split 80/20% (both have 25% 1s). 
    _, training_data, test_data = load_data() # function in experiment_utils.py

    # Make Pytorch Dataloader (batches) 
    test_dataloader = DataLoader(test_data, batch_size=len(test_data)) # one batch for test data 

    X_tw, y_test = next(iter(test_dataloader)) # has weights as last col of X
    X_test = X_tw[:,:-1] # feature vectors
    w_test = X_tw[:, -1] # weights
    
    raw = unjsonify(Path(results_path, 'raw_results.json'))

    meta = unjsonify(Path(results_path, f'experiment_metadata.json'))
    # check that histories agree... they should

    records = [{'AUC': get_auc(torch.tensor(d['beta']), X_test, y_test, w_test)} for d in raw]
    df = pd.DataFrame.from_records(records)
    return df


def plot_99_conf(df, delta = None, label = None, save_fig_path = None, fig_and_ax = None,
                 baseline = None):
    """
    This function plots 99% conf interval for epsilon vs accuracy for the given dataframe.

    Args:
       df (dataframe): experiment results
       name (string): Name at top of title (usually DPSGD or NF)
       delta (float): delta used to compute epsilon (optional)
       label (string): Legend label (default = None)
       save_fig_path (string): where to save the plot (optional)
       fig_and_ax (tuple): Should be the result of plt.subplots(), allows combining plots
    Returns:
       (fig, ax): the resulting plot
    """
    if fig_and_ax == None:fig_and_ax = plt.subplots(layout = "constrained")
    fig, ax = fig_and_ax
    if delta != None:
        label = f"{label}, delta = {delta}"

    sns.regplot(data = df, ax = ax, x = "epsilon", y = "AUC", marker = "_", 
                fit_reg=False, scatter_kws={'alpha':.9, 'linewidths': .5}, label = label,
                line_kws={'alpha':.25}, x_ci = 99, ci = 1, x_bins = 40)

    if type(baseline) != type(None):
        # baseline will have intervals at each epsilon in the range
        med_AUC = baseline.AUC.median()
        ax.axhline(y=med_AUC, color="k", linestyle="--", linewidth = 1)
        ax.text(plt.getp(ax, 'xlim')[0] + .03*np.mean(np.diff(plt.getp(ax, 'xticks'))),
                med_AUC-.3*np.mean(np.diff(plt.getp(ax, 'yticks'))),
                "baseline (no privacy)")

    plt.title(f"Accuracy (AUC) vs. Privacy (epsilon)\n99% Confidence Interval")
    ax.legend()
    if save_fig_path != None:
        fig.savefig(save_fig_path, bbox_inches='tight')
    return fig, ax


def plot_nf_and_dpsgd(nf_df, dpsgd_df, thres = .05, save_fig_path = None,
                      plot_fn = plot_99_conf, baseline = None):
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

    state = (False, -1, 0, 0, math.inf) #(found, prev_ind, curr_ind, prev_closeness, curr_closeness)
    for eps in dpsgd_eps:
        state = (False, state[1], state[1], state[3], math.inf)
        while not state[0] and state[2] <= len(nf_eps)-1:
            d = abs(nf_eps[state[2]] - eps)
            
            if d <= thres:
                if state[1] == state[2]:
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
    
    fig, ax = plt.subplots()
    plot_fn(to_plot_nf, label = 'Exp+NF', fig_and_ax = (fig, ax))
    dpsgd_label = f"DPSGD, noise multi = {dpsgd_df['noise_multiplier'][1]}"
    plot_fn(to_plot_dpsgd, delta, label = dpsgd_label, fig_and_ax = (fig, ax), baseline = baseline)
    ax.set_title("99% Confidence Intervals\nAccuracy (|beta-beta_optimal|) vs. Privacy (epsilon)")
    # fig.text(.5, -0.02, f"DPSGD: delta = {delta}", ha='center')
    ax.legend()
    #fig.show()
    if save_fig_path != None:
        plt.figure(constrained_layout=True)
        fig.savefig(save_fig_path, bbox_inches="tight")
    return fig, ax



if __name__ ==  '__main__': 

######### PROCESS NF results
    # read in resuls dict from expm_results_path
    # %%
    nf_df = get_nf_results(expm_results_path)
    # %%    

    # %% 
    epsilon = 2.0
    meta = unjsonify(Path(expm_metadata_path, f'metadata_{epsilon:.1f}.json'))
    losses = meta['losses']
    log_dets = meta['log_dets']
    utils = meta['utils']
    steps = torch.range(0,len(utils)-1)*1000
 
    # %%
    losses = torch.tensor(losses)
    utils = torch.tensor(utils)
    log_dets = torch.tensor(log_dets)

    # #%%
    with plt.xkcd():
        plt.plot(steps, losses/losses.mean().abs(), label = 'Normalized Loss')
        plt.plot(steps, utils/utils.mean().abs(), label = 'Normalied Ave. Utility')
        plt.plot(steps, log_dets/log_dets.abs().max(), label = 'Normalized Ave. Log Det Jacobian')
        plt.legend()
        plt.xlabel("Training Epoch")
        plt.show()
 
    #%%
    with plt.xkcd():
        plt.plot(steps[:5], losses[:5]/losses.mean().abs(), label = 'Normalized Loss')
        plt.plot(steps[:5], utils[:5]/utils.mean().abs(), label = 'Normalied Ave. Utility')
        plt.plot(steps[:5], log_dets[:5]/log_dets.abs().max(), label = 'Normalized Ave. Log Det Jacobian')
        plt.legend()
        plt.xlabel("Training Epoch")
        plt.show()

    #%% make a plot to save: 
    plt.plot(steps, losses/losses.mean().abs(), label = 'Normalized Loss')
    plt.plot(steps, utils/utils.mean().abs(), label = 'Normalied Ave. Utility')
    plt.plot(steps, log_dets/log_dets.abs().max(), label = 'Normalized Ave. Log Det Jacobian')
    plt.legend()
    plt.xlabel("Training Epoch")
    plt.savefig(Path(expm_metadata_path, 'training_loss.png'))
    plt.savefig(Path(expm_metadata_path, 'training_CIs.svg'), dpi = 700, format = "svg")
    print(f'saved training loss figures in {expm_metadata_path}')
    plt.show()

############# PROCESS DPSGD results
    plot_99_path = Path(plots_path, "99_conf_plots")
    print("Processing DPSGD results")
    dpsgd_df_nm0 = get_dpsgd_results(dpsgd_results_path, 0)
    dpsgd_df_nm1 = get_dpsgd_results(dpsgd_results_path, 1)
    dpsgd_df_nm2 = get_dpsgd_results(dpsgd_results_path, 2)
    dpsgd_df_nm3 = get_dpsgd_results(dpsgd_results_path, 3)

    nonpriv_df = get_non_priv_baseline_results(non_priv_results_path)


    if not plot_99_path.is_dir():plot_99_path.mkdir()
    for with_baseline in [None, nonpriv_df]:

        get_name = lambda s : s if type(with_baseline) == type(None) else f"with_baseline_{s}" 
        plot_99_conf(nf_df, label = "Exp+NF", baseline = with_baseline,
                     save_fig_path = Path(plot_99_path, get_name('expm_99_conf_plot.png')))
        plot_99_conf(dpsgd_df_nm0, label = f"DPSGD, noise multi = {dpsgd_df_nm0['noise_multiplier'][1]}",
                     delta = delta, baseline = with_baseline,
                    save_fig_path = Path(plot_99_path, get_name("dpsgd_nm0_99_conf_plot.png")))
        plot_99_conf(dpsgd_df_nm1, label = f"DPSGD, noise multi = {dpsgd_df_nm1['noise_multiplier'][1]}",
                     delta = delta, baseline = with_baseline,
                    save_fig_path = Path(plot_99_path, get_name("dpsgd_nm1_99_conf_plot.png")))
        plot_99_conf(dpsgd_df_nm2, label = f"DPSGD, noise multi = {dpsgd_df_nm2['noise_multiplier'][1]}",
                     delta = delta, baseline = with_baseline,
                    save_fig_path = Path(plot_99_path, get_name("dpsgd_nm2_99_conf_plot.png")))
        plot_99_conf(dpsgd_df_nm3, label = f"DPSGD, noise multi = {dpsgd_df_nm3['noise_multiplier'][1]}",
                     delta = delta, baseline = with_baseline,
                     save_fig_path = Path(plot_99_path, get_name("dpsgd_nm3_99_conf_plot.png")))

        plot_nf_and_dpsgd(nf_df, dpsgd_df_nm0, baseline = with_baseline,
                          save_fig_path = Path(plot_99_path, get_name("expm_nf_dpsgd_nm0_99_conf_plot.png")))
        plot_nf_and_dpsgd(nf_df, dpsgd_df_nm1, baseline = with_baseline,
                          save_fig_path = Path(plot_99_path, get_name("expm_nf_dpsgd_nm1_99_conf_plot.png")))
        #no overlapping epsilons
        #plot_nf_and_dpsgd(nf_df, dpsgd_df_nm2, save_fig_path = Path(plot_99_path, "expm_nf_dpsgd_nm2_99_conf_plot.png"))
        plot_nf_and_dpsgd(nf_df, dpsgd_df_nm3, baseline = with_baseline,
                          save_fig_path = Path(plot_99_path, get_name("expm_nf_dpsgd_nm3_99_conf_plot.png")))
        print(f'saved confidence interval figures in {plot_99_path}')

# %%
