# plotting utils

import matplotlib.pyplot as plt 
import seaborn as sns
import torch 
import numpy as np 
import pandas as pd
import logging
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)


## see: https://www.kaggle.com/code/daryasikerina/data-visualization-with-seaborn 

# color maps defined: 
virdis = sns.color_palette("viridis", as_cmap=True)
rocket = sns.color_palette("rocket", as_cmap=True) # flare extended to black/white at ends
mako = sns.color_palette("mako", as_cmap=True) # crest extended to black/white at ends
magma = sns.color_palette("magma", as_cmap=True)
inferno = sns.color_palette('inferno', as_cmap = True)
# pretty palettes galore: https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

# Function to sample from base distribution
def random_normal_samples(n, dim=1):
    return torch.zeros(n, dim).normal_(mean=0, std=1)


def _make_grid(min = -4, max = 4, npts = 150):
    """
    Args:
        min (int, optional): min value for x and y axis. Defaults to -4.
        max (int, optional): max value for x and y axis. Defaults to 4.
        npts (int, optional): number of points in each direction. Defaults to 150.

    Returns:
        zgrid = torch.array of shape n^2 x 2. 
            - zgrid[0] = [min, min] (bottom left corner), 
            - zgrid[k] = [min + k epsilon, min] (for k \in [0,n], so it walks along the bottom of the square)
            - zgrid[n] = [min, min + epsilon ]
            - zgrid[n+k] = [min + k, min + epsilon] (so it walks along the second highest row), 
            -...
            ...
    """
    zpoints = torch.linspace(min, max, npts)
    yy, xx = torch.meshgrid(zpoints, zpoints)
    zgrid = torch.vstack([xx.ravel(), yy.ravel()]).T
    return zgrid 


def plot_2D_potential(pot, min = -4, max = 4, npts = 150, cmap = inferno, save_path = None):
    """makes a meshgrid, evaluates and plots the potential function image

    Args:
        pot (_type_): potential function--takes in zgrid ([num_rows, 2], every row gives a point in the plane.)and outputs shape [n_rows] shape but with density/potential value for each row. 
        min (int, optional): x and y axis min. Defaults to -4.
        max (int, optional): x and y axis max. Defaults to -4.
        npts (int, optional): number of points in each direction. Defaults to 150.
        cmap = (cmap variable, optional): colormap variable that must be predefined using seaborn
    """
    zgrid = _make_grid(min, max, npts)
    p = torch.exp(pot(zgrid)) # proportional to probability
    vals = p.reshape(npts, npts).flip(dims = (0,))
    plt.imshow( vals, cmap = cmap, aspect = "equal", extent = [min, max, min, max], interpolation=None )
    plt.colorbar()
    if save_path == None: plt.show()
    else: plt.savefig(save_path)

def plot_pot_func(pot_func, ax=None): # almost the same as above
    if ax is None:
        _, ax = plt.subplots(1)
    x = np.linspace(-4, 4, 100)
    y = np.linspace(-4, 4, 100)
    xx, yy = np.meshgrid(x, y)
    in_tens = torch.Tensor(np.vstack([xx.ravel(), yy.ravel()]).T)
    z = (torch.exp(pot_func(in_tens))).numpy().reshape(xx.shape)

    cmap = plt.get_cmap('inferno')
    ax.contourf(x, y, z.reshape(xx.shape), cmap=cmap)


def plot_1D_potential(pot, min = -4, max = 4, step = None, logplot = True, xkcd = False,  save_path = None): 
    if step == None: step = (max-min)/100
    B = torch.arange(min, max, step).unsqueeze(0)
    if logplot: C = pot(B)
    else: C = torch.exp(pot(B))
    if xkcd: 
        with plt.xkcd():
            plt.plot(B.squeeze(0), C)
            
    else: 
        plt.plot(B.squeeze(0), C, color = inferno.colors[200])

    if logplot: plt.ylabel("log(p_target(x)) + constant")
    else: plt.ylabel("p_target(x) * constant")

    if save_path == None: plt.show()
    else: plt.savefig(save_path)


def plot_loss_v_epochs(losses, skip = 1000, xkcd = True): 
    """Plots losses vs. epoch. 

    Args:
        losses (list): list of training loss values, one per epoch. 
        skip (int, optional): x value step size defaults to 1000, meaning every 1000th point will be plotted. 
    """
    y_losses = losses[0::skip]
    epochs = torch.arange(len(y_losses))*skip
    if xkcd: 
        with plt.xkcd():
            plt.plot(epochs, y_losses) 
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.show()
    else: 
        plt.plot(epochs, y_losses) 
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.show()





def plot_2d_model_and_target(pot, model, N = 1000, min = -4, max = 4, npts=150, cmap = inferno, savepath = None):
    zgrid = _make_grid(min, max, npts)
    p = torch.exp(pot(zgrid)) # proportional to probability
    vals = p.reshape(npts, npts).flip(dims = (0,))
    plt.imshow( vals, cmap = cmap, aspect = "equal", extent = [min, max, min, max], interpolation=None )
    plt.colorbar()
    
    xsamples = random_normal_samples(N, dim = 2)
    zsamples = model.sample(( xsamples )).detach().numpy()

    plt.plot(zsamples[:,0], zsamples[:,1], '.', alpha = .5)
    plt.xlim([min, max])
    plt.ylim([min, max])
    if savepath: plt.savefig(savepath)
    plt.show()

    # kde map: 
    p = sns.jointplot(x = zsamples[:, 0], y = zsamples[:, 1], kind='kde', cmap=cmap)
    p.ax_marg_x.set_xlim(-4, 4)
    p.ax_marg_y.set_ylim(-4, 4)
    if savepath: p.figure.savefig(savepath)
    p.figure.show() 

    # density and samples
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes = axes.flat
    plot_pot_func(pot, axes[0])
    axes[0].set_title('Target density')
    sns.scatterplot(x = zsamples[:, 0], y = zsamples[:, 1], alpha=.2, ax=axes[1])
    axes[1].set_title('Samples')
    if savepath: plt.savefig
    plt.show()

 

def plot_1d_model_and_target(pot, model, min, max,  N = 10000, xkcd = False, save_path = None):
    samples = random_normal_samples(N, dim = 1) # {u_i} sampled from base. 
    transformed_samples = model.sample(samples).detach() # run thru NF 
    dfr = pd.DataFrame(transformed_samples.squeeze(), columns = ["Beta"])

    B = torch.arange(min, max, .001).unsqueeze(0)
    C = torch.exp(pot(B))

    if xkcd: 
        with plt.xkcd():
            fig = plt.figure()
            ax1 = plt.subplot(211)
            ax2 = plt.subplot(212)

            ax1.plot(B.squeeze(0),C)
            ax1.set(ylabel = "p_target * constant")

            ax2 = sns.histplot(data = dfr, x = "Beta")

            ax1.get_shared_x_axes().join(ax1, ax2)
            ax1.set_xticklabels([])

    else: 
        fig = plt.figure()
        ax1 = plt.subplot(211)
        ax2 = plt.subplot(212)

        ax1.plot(B.squeeze(0),C)
        ax1.set(ylabel = "p_target * constant")

        ax2 = sns.histplot(data = dfr, x = "Beta")

        ax1.get_shared_x_axes().join(ax1, ax2)
        ax1.set_xticklabels([])

    if save_path: plt.savefig(save_path) 
    else: plt.show()
