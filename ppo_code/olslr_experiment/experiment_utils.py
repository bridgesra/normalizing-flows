import torch, sys
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd 
from opacus import PrivacyEngine
import math

# %% 
# add our ./code folder to sys.path so we can import our modules: 
sys.path.append(Path(".", "code").absolute().as_posix())
from data_utils import * # our own data utilities
from normalizing_flows import *
from regression_models import *

sys.path.append(Path(".", "code", "olslr_experiment").absolute().as_posix())
from olslr_experiment.config import *

# %% 
# load data
csvpath = Path('.', 'data', 'height-weight.csv' )
if not csvpath.is_file():
    csvpath = Path('..', '..', 'data', 'height-weight.csv')
df = pd.read_csv(csvpath)
X = torch.tensor(df.iloc[:,:-1].values, dtype = torch.float32)
y = torch.tensor(df.iloc[:,1].values, dtype = torch.float32)
if len(X.shape) == 1: # if X is a 1d tensor make 2d
    X.unsqueeze_(1) # shape is now n by 1
# print(f'X.shape = {X.shape}\ny.shape = {y.shape}')


# %% 
# create the utility functions:
def L(B, X = X, y = y): # Loss defined for single vector, b.
    """ Given a matrix B, where each column is a beta vector, 
    for each column (j) of B, this returns  the L2 loss, 
        (1/n) sum_i (x_i^t b[j] - y_i)^2

    Args:
        B (torch.tensor): must be of shape  (k = X.shape[1] by l)--this is l different beta vectors (each are k by 1)
        X (torch.tensor): must be of shape (n by k)        
        y (_type_): must be of shape (n) (vector of scalars)

    Returns:
        (torch.tensor): of shape (l).  j-th row is -(1/n) \sum_: (X[i,:]B[:,j]  - y[:])^2
    """
    c = torch.mm(X,B) # shape is (n by l)
    Y = y.unsqueeze(1) # now shape is (n by 1)
    Y = Y.expand( y.shape[0], B.shape[1]) # shape is (n by l). It copies y to each of the l needed columns. 
    d = c - Y # still n by l. want to return each column d[:,j].pow(2).mean()
    return (d*d).mean(dim = 0)


def util_0(B, X = X, y = y, L = L): # -L
    return - L(B, X, y)


s0 = 22.50384


def util_1(B, X = X, y = y, L = L): # sqrt(-L)
    return -(L(B, X, y)).pow(0.5)


s1 = 2.22 

# loss function for opacus (unfortunately, we can't use util_0 for some reason...)
def L_from_pred(y_pred, y):
    """Given a prediction vector return the L2 loss
       (1/n) sum_i (pred_y_i - y_i)

    Args:
        y_pred (torch.tensor): must be of shape (n x 1)
        y (_type_): must be of shape (n x 1) (vector of scalars)

    Returns:
        (torch.tensor): of shape (l). j-th row is (1/n) \sum_: (y_pred[:] - y[:])^2
    """ 
    d = y_pred - y.unsqueeze(1)
    return (d*d).mean(dim = 0)


# create exponential mechanism distribution's numerator function:
def pot(B, util, epsilon, s):
    """Returns potential function (log of a probability up to a constant) values for each column of B. 
    This function is the log of the numerator of the exponential mechanism distribution: 
    Think of B as a matrix with each column, a "beta". 
    for each column b (= B[:, j]) this returns  pot(b, epsilon, s): = log(exp(util(b,X,y) * epsilon/ (2*s)))


    Args: 
        B (torch.array): of shape k = X.shape[1] by l, l is any positive int]
        util: function of B
        epsilon (float): privacy bound 
        s (float): sensitivity of utility function

    Returns:
        w (torch.array): of shape [l] (a scalar) with potential values for each 1-d point in z. 
    """
    return (util(B) * epsilon) / (2 * s)

# %% 
# model training function: 
def train_model_rms(model, target_density, verbose = True, learning_rate=learning_rate, momentum = momentum, epochs = epochs, batch_size = batch_size):
    """
    Trains model with RMSprop using given lr, momentum, epochs. Uses ReduceLROnPlateau scheduler. 

    Args:
        model (NN class): instantiated model
        target_density(function): lambda x: pot(x.T, util, epsilon, s)--should take in samples = random_normal_samples(batch_size, dim = dim) and produce the potential function output. 
        verbose (bool): if True, prints the loss per epoch. 
    (rest of these args are imported from config.py)    
        learning_rate (float): step size multiplier for optimization
        momentum (float): paramter for how much of last step to mix into this step
        epochs (int): number of mini batches to use in training
        batch_size (int): number of samples to use in each mini batch
    
    Returns:
        list of floats: losses of each epoch.  
    """
    dim = model.dim
    
    # RMSprop is what they used in renzende et al
    opt = torch.optim.RMSprop(
        params = model.parameters(),
        lr = learning_rate,
        momentum = momentum
    )

    scheduler = ReduceLROnPlateau(opt, 'min', patience=1000) # hmmm, looks interesting 
    
    losses = [] # to be populated

    for epoch in range(epochs):
        if verbose and (epoch % 1000) == 0:
            print(f"Epoch {epoch}")

        # samples = torch.autograd.Variable(random_normal_samples(batch_size)) # Variables is deprecated and i think this is unnecessary
        samples = random_normal_samples(batch_size, dim = dim) # {u_i} sampled from base. 

        # compute  RKL loss function: 
        x, log_det_T = model(samples) # we need the log_det_T and the x_i = T(u_i) 
        log_p_x = target_density(x) # log(p*(x_i)), needed for loss function
        loss = -(log_det_T + log_p_x).mean() # Reverse KL
        # if we add the log_p_u, and we tell pytorch to track the params of the base model, and pass the args to an optimizer, can we optimize both base model and flow? 

        # take a step: 
        opt.zero_grad()

        loss.backward()
        opt.step()
        # scheduler.step(loss)

        losses.append(loss.item())

        if verbose and (epoch % 1000) == 0:
            print("Loss {}".format(loss.item()))
    return losses 



def compute_epsilon(epoch, delta, noise_multiplier, sample_rate, steps_per_epoch, eps_error = .01):
    """
    This computes epsilon using the PRV Accounting method for the given delta at the specified epoch with the information provided in the results.
    Args:
        epoch (int): the epoch to compute epsilon for
        delta (float): delta for privacy
        noise_multiplier (float): from results
        sample_rate (float): from results
        steps_per_epoch (int): number of calls to opt.step per epoch
        eps_error (float): allowable error for computed epsilon
    Returns:
        float: computed epsilon
    """
    privacy_engine = PrivacyEngine(accountant = 'prv')
    epoch_steps = epoch * steps_per_epoch + steps_per_epoch
    privacy_engine.accountant.history.append((noise_multiplier, sample_rate, epoch_steps))
    return privacy_engine.accountant.get_epsilon(delta, eps_error = eps_error)


# model training function:
# since we want several epsilons which are computed iteratively, we will save the model in this function
def train_dpsgd_model(model, dataloader, loss, opt, verbose = True, save_state = False, noise_multiplier = dpsgd_noise_multiplier0, epochs = epochs, metadata_path = None):
    """
    Trains model with RMSprop using given lr, momentum, epochs. Uses ReduceLROnPlateau scheduler. 

    Args:
        model (NN class): instantiated model
        dataloader (torch.DataLoader): data that the model will train on, split into batches
        verbose (bool): if True, prints the loss per epoch.
        save_state (bool): if True, save the model, opacus privacy engine and optimizer
    (rest of these args are imported from config.py)    
        epochs (int): number of mini batches to use in training
        noise_multiplier (float): how much noise to add when privately training
    
    Returns:
        list of floats: losses of each epoch
        list of floats: betas
        dict: accountant history for computing epsilon later
    """
    
    # Make private with Opacus
    privacy_engine = PrivacyEngine()
    steps_per_epoch = 0
    sample_rate = 1/len(dataloader)

    priv_model, priv_opt, priv_dataloader = privacy_engine.make_private(
    module=model,
    optimizer=opt,
    data_loader=dataloader,
    noise_multiplier=noise_multiplier,
    max_grad_norm=1.0,
    )

    # to be populated
    losses = []
    betas = [] 

    for epoch in range(dpsgd_epochs):
        if verbose and (epoch % 10) == 0:
            print(f"Epoch {epoch}")
        for batched_X, batched_y in dataloader:
            # clear gradients (must be first or opacus complains)
            priv_opt.zero_grad()
            pred_y = priv_model(batched_X)

            # compute  loss
            l = loss(pred_y, batched_y)

            # take a step: 
            l.backward()
            priv_opt.step()
            if epoch == 0:
                # determine steps per epoch so we can compute privacy later
                steps_per_epoch += 1

        losses.append(l.item())
        betas.append(priv_model.linear.weight.item()) # learned beta

        # saving privacy state
        # we'll just compute epsilon each epoch and save that.
        # https://discuss.pytorch.org/t/how-to-store-the-state-and-resume-the-state-of-the-privacyengine/138538
        if save_state:
            torch.save(privacy_engine.accountant, Path(metadata_path, f'privacy_accountant_eps_{epsilon}.pth').as_posix())
            torch.save(priv_model._module.state_dict(), Path(metadata_path, f'privacy_accountant_eps_{epsilon}.pth').as_posix())
            torch.save(priv_opt._module.state_dict(), Path(metadata_path, f'privacy_accountant_eps_{epsilon}.pth').as_posix())

        if verbose and (epoch % 10) == 0:
            print("Loss {}".format(l.item()))
        if math.isnan(l):
            break
    return losses, betas, {"noise_multiplier" : noise_multiplier, "sample_rate" : sample_rate, "steps_per_epoch" : steps_per_epoch}



def kernel_fn(util_index, util, s, epsilon,  metadata_path, num_samples = 1000, pot = pot,  n_flows = n_flows, training_fn = train_model_rms, learning_rate=learning_rate, momentum = momentum, epochs = epochs, batch_size = batch_size): 
    """ This is the kernel function to be passed to multiprocess.Pool().starmap() when running experiment 
        Given the utility function, its index, privacy bound epsilon, and the metadath_path, 
        this trains a planar NF model to approximate the potential function and samples num_samples (betas) from it. 
        It will save the model and losses in metadata_path 
        It will return a dict with the samples (betas), and it's info (epsilon, util_index, etc. )

    Args:
        util_index (int): 0 or 1, index of which utility function is being passed
        util(function): utility function defined in experiment_utils.py (eventually to be optimized via exponential mechanism sampling) 
        s (float): sensitivity of the utility function
        epsilon (float): privacy bound parameter, must be positive
        metadata_path (str): path to results subfolder for where to store the model and losses during training
        ---
        num_samples (int, optional): number of samples to draw from the trained model. Defaults to 1000.
        pot (function, optional): potential function defined in experiment_utils.py (the log(numerator(expm density)))
        n_flows (int, optional): number of layers in the NF to use. Defaults to n_flows as defined/imported in config.py.
        training_fn(function, optional): torch training function, see train_model_rms() above
    (rest of these args are training parameters, imported from config.py)    
        learning_rate (float): step size multiplier for optimization
        momentum (float): paramter for how much of last step to mix into this step
        epochs (int, optional): numer of epochs to use in training. Defaults to epochs as defined/imported in config.py.
        batch_size (int, optional): number of samples used in each batch Defaults to batch_size  as defined/imported in config.py.
        
    Returns:                    # 
        (dict): results dict {util_index, epsilon, sampled_betas}
    """
    global dim
    # print(f"\tBeginning epsilon = {epsilon}")
    results = {'util': util_index, 'epsilon':  epsilon}
    # instantiate then train model model:
    model = NormalizingFlow(dim, n_flows=n_flows) # number of layers/flows = n_flows

    def target_density(x): return pot(x.T, util=util, epsilon=epsilon, s=s)  # needed for training      
    losses = training_fn(model, target_density, verbose= False, learning_rate=learning_rate, momentum = momentum, epochs = epochs, batch_size = batch_size )  # trains model

    # save model metadata
    torch.save(model, Path(metadata_path, f'model_eps_{epsilon}.pth').as_posix())
    jsonify(losses, Path(metadata_path, f'losses_eps_{epsilon}.json').as_posix())

    # sample num_samples betas from the NF model:
    samples = random_normal_samples(num_samples, dim=1) # {u_i} sampled from base.
    transformed_samples = model.sample(samples).detach() # runs base samples thru NF
    results['betas'] = transformed_samples.squeeze().tolist()  # puts samples in list
    return results



# model training function:
# since we want several epsilons which are computed iteratively, we will save the model in this function
def train_dpsgd_model_rms(model, dataloader, loss, verbose = True, save_state = False, learning_rate=dpsgd_learning_rate, momentum = momentum, epochs = dpsgd_epochs, batch_size = batch_size, delta = delta, metadata_path = None):
    """
    Trains model with RMSprop using given lr, momentum, epochs. Uses ReduceLROnPlateau scheduler. 

    Args:
        model (NN class): instantiated model
        dataloader (torch.DataLoader): data that the model will train on, split into batches
        verbose (bool): if True, prints the loss per epoch.
        save_state (bool): if True, save the model, opacus privacy engine and optimizer
    (rest of these args are imported from config.py)    
        learning_rate (float): step size multiplier for optimization
        momentum (float): paramter for how much of last step to mix into this step
        epochs (int): number of mini batches to use in training
        batch_size (int): number of samples to use in each mini batch
    
    Returns:
        losses: losses of each epoch.
        epsilons: epsilons of each epoch.
        betas: beta for each epoch.
    """
    
    # RMSprop is what we used above
    opt = torch.optim.RMSprop(
        params = model.parameters(),
        lr = learning_rate,
        momentum = momentum
    )
    # Make private with Opacus
    privacy_engine = PrivacyEngine()
    priv_model, priv_opt, priv_dataloader = privacy_engine.make_private(
    module=model,
    optimizer=opt,
    data_loader=dataloader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
    )


    # to be populated
    losses = []
    epsilons = [] 
    betas = [] 

    for epoch in range(epochs):
        if verbose and (epoch % 10) == 0:
            print(f"Epoch {epoch}")
        for batched_X, batched_y in dataloader:
            # clear gradients (must be first or opacus complains)
            priv_opt.zero_grad()
            pred_y = priv_model(batched_X)

            # compute  loss
            l = loss(pred_y, batched_y)

            # take a step: 
            l.backward()
            priv_opt.step()

        losses.append(l.item())
        betas.append(priv_model.linear.weight.item()) # learned beta

        epsilon = privacy_engine.accountant.get_epsilon(delta)
        epsilons.append(epsilon)
        # saving privacy state
        # we'll just compute epsilon each epoch and save that.
        # https://discuss.pytorch.org/t/how-to-store-the-state-and-resume-the-state-of-the-privacyengine/138538
        if save_state:
            torch.save(privacy_engine.accountant, Path(metadata_path, f'privacy_accountant_eps_{epsilon}.pth').as_posix())
            torch.save(priv_model._module.state_dict(), Path(metadata_path, f'privacy_accountant_eps_{epsilon}.pth').as_posix())
            torch.save(priv_opt._module.state_dict(), Path(metadata_path, f'privacy_accountant_eps_{epsilon}.pth').as_posix())

        if verbose and (epoch % 10) == 0:
            print("Loss {}".format(l.item()))
            print("Epsilon {}".format(epsilon))
    return losses, epsilons, betas




def dpsgd_kernel_fn(model_index, metadata_path, training_fn = train_dpsgd_model_rms, learning_rate=learning_rate, momentum = momentum, epochs = epochs, batch_size = batch_size, delta = delta): 
    """ This is the kernel function to be passed to multiprocess.Pool().starmap() when running experiment 
        Given the model index, noise_multiplier, and the metadath_path, 
        this trains a DPSGD  model to approximate beta at various epsilons
        It will save the losses in metadata_path 
        It will return a dict with the betas (the model) at the corresponding epsilon, and it's info (model_index, etc. )

    Args:
        model_index (int): index of the model being trained (it's initalized in this function)
        noise_multiplier (float): how much noise to use during training 
        metadata_path (str): path to results subfolder for where to store the losses during training
        ---
        training_fn(function, optional): torch training function, see train_dpsgd_model_rms() above
    (rest of these args are training parameters, imported from config.py)    
        learning_rate (float): step size multiplier for optimization
        momentum (float): paramter for how much of last step to mix into this step
        alpha (float): smoothing parameter in RMSProp
        epochs (int, optional): numer of epochs to use in training. Defaults to epochs as defined/imported in config.py.
        batch_size (int, optional): number of samples used in each batch Defaults to batch_size  as defined/imported in config.py.
        
    Returns:
        (dict): results dict {util_index, beta for each epoch, accountant history for computing privacy later}
    """
    results = {'model_index' : model_index}
    print("Model index {}".format(model_index))
    # instantiate then train model model:
    model = LinearRegressionModel()
    opt = torch.optim.RMSprop(
        params = model.parameters(),
        lr = learning_rate,
        alpha = alpha,
        momentum = momentum
    )

    # saving all these models seems excessive... 
    dataloader = DataLoader(dataset = DatasetMaker(X,y), batch_size = batch_size, shuffle = True)  # 
    losses, betas, accountant_history = training_fn(model, dataloader, L_from_pred, opt, verbose = True, epochs = epochs, save_state = False, metadata_path = metadata_path, noise_multiplier = noise_multiplier)

    # save model metadata
    jsonify(losses, Path(metadata_path, f'losses_model_{model_index}.json').as_posix())

    results['betas'] = betas
    results['accountant_history'] = accountant_history
    return results


def dpsgd_hyperparameter_search_kernel_fn(model_index, util_index, util, param_index, param_dict, metadata_path, epochs, batch_size, train_fn = train_dpsgd_model): 
    """ This is the kernel function to be passed to multiprocess.Pool().starmap() when running hyperparameter experiment
        Given the model index, utility function, its index, and hyperparameters, and the metadath_path, 
        this trains a DPSGD  model to approximate beta
        It will save the losses in metadata_path
        It will return a dict with the betas (the model) at each epoch, and it's info (util_index, model_index, etc. )

    Args:
        model_index (int): index of the model being trained (it's initalized in this function)
        util_index (int): 0 or 1, index of which utility function is being passed
        util(function): utility function defined in experiment_utils.py
        param_dict (dict): hyperparameters for RMSProp optimizer
        metadata_path (str): path to results subfolder for where to store the losses during training
        ---
        training_fn(function, optional): torch training function, see train_dpsgd_model_rms() above
    (rest of these args are training parameters, imported from config.py)    
        epochs (int, optional): numer of epochs to use in training. Defaults to epochs as defined/imported in config.py.
        
    Returns:
        (dict): results dict {util_index, beta for each epoch, accountant history for computing privacy later}
    """
    for_saving_param_dict = dict(**param_dict, batch_size = batch_size)
    results = {'param_index': param_index, 'hyperparameters': for_saving_param_dict, 'model_index' : model_index, 'util': util_index}
    print("Model index {}, util index {}".format(model_index, util_index))
    # instantiate then train model model:
    model = LinearRegressionModel()
    opt = torch.optim.RMSprop(
        params = model.parameters(),
        **param_dict
    )

    # saving all these models seems excessive... 
    dataloader = DataLoader(dataset = DatasetMaker(X,y), batch_size = batch_size, shuffle = True)  # 
    losses, betas, accountant_history = train_fn(model, dataloader, util, opt, verbose = True, epochs = epochs, save_state = False, metadata_path = metadata_path)

    # save model metadata
    jsonify(losses, Path(metadata_path, f'param_{param_index}_losses_model_{model_index}.json').as_posix())

    results['betas'] = betas
    results['accountant_history'] = accountant_history
    return results



