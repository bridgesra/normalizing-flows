import pandas as pd 
from pathlib import Path
import sys 
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader
from opacus import PrivacyEngine
# from opacus.validators import ModuleValidator
from sklearn import metrics 
import torch, sys
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.append(Path(".", "code").absolute().as_posix())
sys.path.append(Path("..","..", "code").absolute().as_posix())
sys.path.append(Path(".", "code", "lr_experiment").absolute().as_posix())
from data_utils import *
from regression_models import LogisticRegression
from normalizing_flows import *
from config import *

def normalize_data(X, norm = ''): 
	"""
	if norm = '', this just returns X. 
	If norm = 'l1', 'l2', or 'linf' this replaces the numerical columns (including fnlwgt col) with x[:,j]/||x[:,j]||_1,2,\infty, respectively
	NOTE it is given that column `age` has minimum 17, so for this column we first subtract 17, then divide by the resulting norm. 
	Args:
		X (dataframe): Adult dataset features matrix
		norm (str, optional): '', 'l1', or 'l2'. Defaults to ''.

	Returns:
		dataframe: same as X but values in numerical columns normalized
	"""
	if norm == '': return X
	cols = [col for col in X.columns if len(X[col].unique())>2 ]
	A = X.copy()
	A = A.rename({c: c+'_raw' for c in  cols}, axis = 1)
	A['age_raw']-= 17    
	for c in cols:       
		if norm == 'linf': 
			A[c] = A[ c + "_raw" ] / A[c+'_raw'].abs().max()            
		elif norm == 'l1': 
			A[c] = A[ c + "_raw" ] / A[c+'_raw'].abs().sum()
		else: # l2
			A[c] = A[ c + "_raw" ] / ( (A[c+'_raw']**2).sum() )**(.5)
	
	return A[X.columns]


def load_data(test_size = .2, verbose = True):
	"""reads in cleaned adult dataset and returns dataframe, 
		also makes and returns torch.Dataset objects training_data, testing_data with split percent passed

	Args:
		test_size (float, optional): percent for testing. Defaults to .2.
		verbose (bool, optional): flag for printoffs. Defaults to True.

	Returns:
		triple: pandas dataframe, training data (torch.Dataset), testing data (torch.Dataset)
	"""
	if verbose: print('Loading Adult Dataset ...')
	df = pd.read_csv(ADULT_CLEAN_FULL_PATH, index_col= 0) # read in data
	X_w = df[df.columns.difference(['target'])] # remove target to make X matrix with weights column

	# moving weights column, fnlwgt to last column
	cols = X_w.columns.difference(['fnlwgt']).to_list() + ['fnlwgt']
	X_w = X_w[ cols ]
	y = df.target

	if verbose: print(f'Making stratified {1-test_size}/{test_size}% train/test split ...')
	# make stratified train/test split, that is, with 25/75% 1/0s in both train and test: 
	sss = StratifiedShuffleSplit(n_splits = 1, test_size=test_size, random_state = 0)
	train_index, test_index = sss.split( X_w, y).__next__() # only 1 object in this generator 

	X_train_w = normalize_data(X_w.iloc[train_index], 'linf')
	y_train = y.iloc[train_index]
	training_data = DatasetMaker(X_train_w, y_train)

	X_test_w = normalize_data(X_w.iloc[test_index], 'linf')
	y_test =  y.iloc[test_index]
	test_data = DatasetMaker(X_test_w, y_test)
		
	if verbose: 
		print(f"Training set has {y_train.sum()/ len(y_train)} 1's and the test set has {y_test.sum()/len(y_test)} 1's")
	
	return df, training_data, test_data 


# loss function is the binary cross entropy + L1 regularization. 
def loss_bce(model, X, y, weight = None, c = 1):
    """returns -\sum_i w_i [ y_i \log (\phi(x_i*beta)) + (1-y_i)\log(1-\phi(x_i*beta)) ] + c ||beta||_1

    Args:
        model (torch.nn.Model): Logistic regression model (or any binary classification model with outputs in (0,1))
        X (torch.tensor): one batch's design matrix. Shape is (batch size, number of features). each row is a feature vector
        y (torch.tensor): one batch's list of targets. Shape is (batch size, ). entries are in {0,1}
        weight (torch.tensor): one batch's list of weights. Shape is (batch size, ). Optional. Default = None
        c (float): regularization parameter. Optional. Default = 1
    Returns:
        _type_: _description_
    """
    # first compute cross entropy loss:
    cel = torch.nn.BCELoss(weight = weight, reduction = 'sum' ) ## instantiates binary cross entropy loss function class, see # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    y_pred = model.forward(X).squeeze() # has one extra, unused dimension we need to squeeze
    a = cel(y_pred, y)

    # next compute regularization: 
    p = list(model.parameters()) # two element list. first is a list of all the non bias parameters, second element is a list of the lone bias parameter
    x = p[0].squeeze().abs().sum() # sum of abs of all but bias parameter
    reg = x + p[1].abs() #sum of abs of all parameters, the l1 norm. 
    
    return (a + c*reg)

def loss_l2(model, X, y, weight = None, c = .1):
    y_preds = model.forward(X).squeeze()
    loss = weight*(y_preds - y).pow(2)
    loss = loss.sum()
    p = list(model.parameters())
    reg = p[0].squeeze().abs().sum() + p[1].abs()

    return loss + c*reg

# ExpM+NF functions
def util(B, X, y, w,  c = nf_c ):
	"""
	utility funciton to be maximized. returns -sum(ypred-y)^2 -c |beta|_1
	Args:
		B (torch.tensor): must be of shape  (l by k+1 = X.shape[1]+1)--this is l different beta vectors (each are k by 1)        
		X (torch.tensor): must be shape (n by k). training feature matrix. 
		y (torch.tensor, values in {0,1}): must be shape (n), training targets. 
		w (torch.tensor, positive values): must be shape (n), training weights. 
		c (float, optional): regularization parameter. Defaults to 0.1.

	Returns:
		torch.tensor: shape is (l), utility
	"""
	X_1 = torch.concat([X, torch.ones([X.shape[0], 1]) ], axis = 1) # adds 1s for last column 
	assert B.shape[1] == X_1.shape[1]

	# negative of L2 loss
	y_preds = torch.sigmoid(torch.mm(X_1,B.T)) # shape is (2 by l): \phi(x^t b)  for each of the 2 data points x\in X, and for each of the l weight vectors b\in B, 
	yl = torch.stack([y ]*y_preds.shape[1], 1) # make labels for every y_pred value
	losses = (y_preds - yl).pow(2) # shape is n by l (loss of all n data points for each of l betas)
	wl = torch.stack([w]*losses.shape[1], 1) # make labels for every losses value
	losses = losses * wl # times weight
	losses = losses.sum(0) #sum down the rows, should have shape [l]

	#regularize: compute l2 norm^2 of input
	# B_1, B_2 = B[:, 0], B[:, 1]
	# regs = B_1.pow(2) + B_2.pow(2)
	
	# compute l1 norm of every beta given. 
	regs = B.abs().sum(axis = 1) # shape is l, the l1 norm of each of the l betas
   
	return -losses - c*regs


def pot(B, X, y, w, epsilon, s = s,  util = util, c = nf_c):
	"""Returns potential function (log of a probability up to a constant) values for each column of B. 
	This function is the log of the numerator of the exponential mechanism distribution: 
	Think of B as a matrix with each column, a "beta". 
	for each column b (= B[:, j]) this returns  pot(b, epsilon, s): = log(exp(util(b,X,y) * epsilon/ (2*s))) = util(b,X,y)*epsilon/2s

	Args: 
		B (torch.tensor): must be of shape l by k+1 (k= X.shape[1])--this is l different beta vectors (each are k by 1)        
		epsilon (float): privacy bound 
		s (float): sensitivity of utility function
		util (function): function of B (see above). Defaults to util above
		X (torch.tensor): must be shape (n by k). training feature matrix. should be X_train.
		y (torch.tensor, values in {0,1}): must be shape (n), training targets. should be y_train.
		w (torch.tensor, positive values): must be shape (n), training weights. should be w_train.
		c (float, optional): regularization parameter. Defaults to 0.1.

	Returns:
		v (torch.tensor): of shape l = B.shape[0]--the potential values for each beta (each row of B)
	"""
	return (util(B,  X = X, y = y, w = w, c = c) * epsilon) / (2 * s)


# model training function: 
def train_model_reverseKL(model, target_density,  X, y, w, util = util, c = nf_c, verbose = True, \
	learning_rate=nf_learning_rate, momentum = nf_momentum, epochs = nf_epochs, batch_size = nf_batch_size):
	"""
	Trains model with RMSprop using given lr, momentum, epochs. Uses ReduceLROnPlateau scheduler. 

	Args:
		model (NN class): instantiated model
		target_density(function): lambda x: pot(x.T, util, epsilon, s)--should take in samples = random_normal_samples(batch_size, dim = dim) and produce the potential function output. 
		util (function) : utility function 
  		X (torch.tensor): must be shape (n by k). training feature matrix. Defaults to X_train.
		y (torch.tensor, values in {0,1}): must be shape (n), training targets. Defaults to y_train.
		w (torch.tensor, positive values): must be shape (n), training weights. Defaults to w_train.
		c (float, optional): regularization parameter. Defaults to 0.1.
  		verbose (bool): if True, prints the loss per epoch. 
	(rest of these args are imported from config.py)    
		learning_rate (float): step size multiplier for optimization
		momentum (float): paramter for how much of last step to mix into this step
		epochs (int): number of mini batches to use in training
		batch_size (int): number of samples to use in each mini batch
	
	Returns:
		quadruple: losses, utils, log_dets (all lists), model (normalizing flow model)
	"""
	dim = model.dim
	
	# RMSprop is what they used in renzende et al
	opt = torch.optim.RMSprop(\
		params = model.parameters(),\
		lr = learning_rate,\
		momentum = momentum\
		)
	
	scheduler = ReduceLROnPlateau(opt, 'min', patience=1000) 

	# to be populated
	steps = []
	losses = [] 
	utils = [] 
	log_dets = []

	skip = 1000 # how often in terms of epochs to record loss
	for epoch in range(epochs):
		samples = random_normal_samples(batch_size, dim = dim) # {u_i} sampled from base. 

		# compute  RKL loss function: 
		x, log_det_T = model(samples) # we need the log_det_T and the x_i = T(u_i) 
		log_p_x = target_density(x).unsqueeze(1) # log(p*(x_i)), needed for loss function
		loss = -(log_det_T + log_p_x).mean() # Reverse KL

		# Store loss values before stepping: 
		if epoch % skip == 0:        
			with torch.no_grad():
				steps.append(epoch)
				utils.append(util(x, X, y, w, c = c).mean().item())
				log_dets.append(log_det_T.mean().item()) 
				losses.append(loss.item())
			if verbose: print(f"Epoch {epoch}\t Loss {loss.item()}\t Ave. Util {utils[-1]}\t Log Det Jac {log_dets[-1]}")

		# take a step: 
		opt.zero_grad()
		loss.backward()
		opt.step()
		scheduler.step(loss)
	return losses, utils, log_dets, model 


def kernel_fn(epsilon, X, y, w, \
	metadata_path = expm_metadata_path, num_samples = 1000, s = s, util = util, pot = pot, \
	n_flows = n_flows, training_fn = train_model_reverseKL, \
	learning_rate = nf_learning_rate, momentum = nf_momentum, epochs = nf_epochs, batch_size = nf_batch_size): 
	""" This is the kernel function to be passed to multiprocess.Pool().starmap() when running experiment 
		This trains a planar NF model to approximate the potential function and samples num_samples (betas) from it. 
		It will save losses, utility function values, and log det jacobian values in metadata_path 
		It will return a dict with the samples (betas), and it's info (epsilon, util_index, etc. )

	Args:
		epsilon (float): privacy bound parameter, must be positive
		X (torch.tensor): must be shape (n by k). training feature matrix. Defaults to X_train.
		y (torch.tensor, values in {0,1}): must be shape (n), training targets. Defaults to y_train.
		w (torch.tensor, positive values): must be shape (n), training weights. Defaults to w_train.
		---
		metadata_path (str): path to results subfolder for where to store the model and losses during training (defined in config)
		num_samples (int, optional): number of samples to draw from the trained model. Defaults to 1000.
		s (float): sensitivity of the utility function, default is 2 as defined in config.
		util(function): utility function defined in experiment_utils.py (eventually to be optimized via exponential mechanism sampling) 
		pot (function, optional): potential function defined in experiment_utils.py (the log(numerator(expm density)))
		n_flows (int, optional): number of layers in the NF to use. Defaults to n_flows as defined/imported in config.py.
		training_fn(function, optional): torch training function, see train_model_reverseKL() above
	(rest of these args are training parameters, imported from config.py)    
		learning_rate (float): step size multiplier for optimization
		momentum (float): paramter for how much of last step to mix into this step
		epochs (int, optional): numer of epochs to use in training. Defaults to epochs as defined/imported in config.py.
		batch_size (int, optional): number of samples used in each batch Defaults to batch_size  as defined/imported in config.py.
		
	Returns:
		(dict): results dict {epsilon, sampled_betas}
	"""
	dim = X.shape[1] + 1 # plus 1 for bias term 

	# instantiate then train model model:
	model = NormalizingFlow(dim, n_flows=n_flows) # number of layers/flows = n_flows
	target_density = lambda B: pot(B, X, y, w, epsilon, s = s,  util = util, c = c)
 
	losses, utils, log_dets, model = training_fn(model, target_density, X, y, w, util = util, c =c,  verbose = False, \
		learning_rate = learning_rate, momentum = momentum, epochs = epochs, batch_size = batch_size)

	# save metadata
	metadata = {'losses': losses, 'utils': utils, 'log_dets': log_dets}
	# save model metadata
	jsonify(metadata, Path(metadata_path, f'metadata_{epsilon}.json').as_posix())

	# sample num_samples betas from the NF model:
	samples = random_normal_samples(num_samples, dim = dim) # {u_i} sampled from base.
	transformed_samples = model.sample(samples).detach() # runs base samples thru NF
	# return results: 
	results = {}
	results['epsilon'] = epsilon
	results['betas'] = transformed_samples.squeeze().tolist()  # puts samples in list
	return results


def get_auc(b, X_test, y_test, w_test):
	"""Used to evaluate NF samples. given b (tensor of shape [X.shape[1] + 1])
		this instantiates a logistic regrssion model with b[-1] as the bias term 
		then computes the area under the ROC for the given data. 

	Args:
		b (torch.tensor): _description
		X_test (torch.tensor): _description_. Defaults to X_test.
		y_test (torch.tensor): _description_. Defaults to y_test.
		w_test (torch.tensor): _description_. Defaults to w_test.

	Returns:
		float: roc_auc 
	"""
	input_dim = X_test.shape[1] #  (already has the weights column removed)
	output_dim = 1
	lr_model = LogisticRegression(input_dim, output_dim)
	state_dict = {
		'linear.weight' : b[:-1].reshape([1,102]),
		'linear.bias' : b[-1].reshape([1])
		}
	lr_model.load_state_dict(state_dict)

	with torch.no_grad():
		y_pred = lr_model.forward(X_test).squeeze().detach()

	targets = y_test.detach()

	return metrics.roc_auc_score(targets, y_pred,  sample_weight=w_test)


#DPSGD utils: 
def train_dpsgd_model_rms(model, train_dataloader, test_dataloader, loss = loss_l2,
                          verbose = True, save_state = False, learning_rate=dpsgd_learning_rate,
                          momentum = dpsgd_momentum, epochs = dpsgd_epochs,
                          metadata_path = None, noise_multiplier = dpsgd_noise_multiplier0):
    """
    Trains a private model with PRVAccountant using the given optimizer (with weights).

    Args:
        model (NN class): instantiated model
        train_dataloader (torch.DataLoader): data that the model will train on, split into batches
        test_dataloader (torch.DataLoader): data that the model will test on
        verbose (bool): if True, prints the loss and epsilon (using delta) per epoch.
        save_state (bool): if True, save the model, opacuss privacy engine and optimizer
    (rest of these args are imported from config.py)
        epochs (int): number of iterations over entire dataset
        learning_rate (float): step size multiplier for optimization
        momentum (float): paramter for how much of last step to mix into this step
        noise_multiplier (float): how much noise to use during training

    Returns:
        train_loss: list of losses of training set each epoch
        betas: list of beta at each epoch
        accountant_dict: a dictionary of privacy accounting information necessary to compute epsilon later
    """

  # Make private with Opacus
    privacy_engine = PrivacyEngine(accountant = 'prv')
    opt = torch.optim.RMSprop(
        params = model.parameters(),
        lr = learning_rate,
        momentum = momentum
    )
    priv_model, priv_opt, priv_dataloader = privacy_engine.make_private(
    module=model,
    optimizer=opt,
    data_loader=train_dataloader,
    noise_multiplier=noise_multiplier,
    max_grad_norm=1.0)

    # to be populated
    train_losses = []
    sample_rate = 1/len(train_dataloader) # sample rate used in accountant, this is the default
    total_steps = 0
    steps_per_epoch = 0
    betas = []

    # make test set tensors: 
    X_tw, y_t = next(iter(test_dataloader))
    y_target = y_t.detach()
    X_t = X_tw[:,:-1] # feature vectors
    w_t = X_tw[:, -1] # weights

    for epoch in range(epochs): # for each epoch
        for Xw, y_ in priv_dataloader: #for each batch   

            X_ = Xw[:, :-1] # all but last/weights column
            w_ = Xw[:,-1] # only weights column 

            l = loss(priv_model, X_, y_, weight = w_, c = dpsgd_c)
            # take a step:
            priv_opt.zero_grad()
            total_steps += 1 # needed for computing privacy later
            l.backward()
            priv_opt.step()
            if epoch == 0:
                # determine steps per epoch so we can compute privacy later
                steps_per_epoch += 1

        # this allows us to compute privacy later with a variety of deltas if we so choose
        train_losses.append(l.item())  #
        beta = priv_model.linear.weight.detach().squeeze().tolist() + priv_model.linear.bias.detach().tolist()
        betas.append(beta)

        # saving privacy state
        # https://discuss.pytorch.org/t/how-to-store-the-state-and-resume-the-state-of-the-privacyengine/138538
        if save_state:
            torch.save(privacy_engine.accountant, Path(metadata_path, f'privacy_accountant_epoch_{epoch}.pth').as_posix())
            torch.save(priv_model._module.state_dict(), Path(metadata_path, f'privacy_accountant_epoch_{epoch}.pth').as_posix())
            torch.save(priv_opt._module.state_dict(), Path(metadata_path, f'privacy_accountant_epoch_{epoch}.pth').as_posix())

        if verbose and epoch % 25 == 0:
            # now record losses on train and test: 
            print(f"Epoch {epoch}")
            print("\tTraininxg Loss {}".format(l.item()))

    return train_losses, betas, {"noise_multiplier" : noise_multiplier, "sample_rate" : sample_rate, "steps_per_epoch" : steps_per_epoch}


def compute_epsilon(epoch, delta, noise_multiplier, sample_rate, steps_per_epoch,
                    eps_error = .01, accountant = 'prv'):
    """
    This computes epsilon using the provided accounting method for the given delta at the specified epoch with the information provided in the results.
    Args:
        epoch (int): the epoch to compute epsilon for
        delta (float): delta for privacy
        noise_multiplier (float): from results
        sample_rate (float): from results
        steps_per_epoch (in): number of calls to opt.step per epoch
        eps_error (float): error when computing epsilon with PRV accounting method (default = .01)
    """
    privacy_engine = PrivacyEngine(accountant = accountant)
    epoch_steps = epoch * steps_per_epoch + steps_per_epoch
    privacy_engine.accountant.history.append((noise_multiplier, sample_rate, epoch_steps))
    if accountant == 'prv':
        return privacy_engine.accountant.get_epsilon(delta, eps_error = eps_error)
    else:
        return privacy_engine.accountant.get_epsilon(delta)


def dpsgd_kernel_fn(model_index, noise_multiplier, metadata_path, loss = loss_l2, train_fn = train_dpsgd_model_rms, learning_rate=dpsgd_learning_rate, momentum = dpsgd_momentum, epochs = dpsgd_epochs, batch_size = dpsgd_batch_size): 
    """ This is the kernel function to be passed to multiprocess.Pool().starmap() when running experiment 
        Given the model index, utility function, its index, privacy bound epsilon, and the metadath_path, 
        this trains a DPSGD  model to approximate beta at various epsilons measured during training
        It will save the losses in metadata_path 
        It will return a dict with the betas (the model) at the corresponding epsilon, and it's info (util_index, model_index, etc. )

    Args:
        model_index (int): index of the model being trained (it's initalized in this function)
        noise_multiplier (float): how much noise to use during training
        metadata_path (str): path to results subfolder for where to store the losses during training
        ---
        training_fn (function, optional): torch training function, see train_dpsgd_model_rms() above
        loss (function, option): a loss function, see loss_fu above
    (rest of these args are training parameters, imported from config.py)    
        learning_rate (float): step size multiplier for optimization
        momentum (float): paramter for how much of last step to mix into this step
        epochs (int, optional): numer of epochs to use in training. Defaults to epochs as defined/imported in config.py.
        batch_size (int, optional): number of samples used in each batch Defaults to batch_size  as defined/imported in config.py.
        
    Returns:
        (dict): results dict {model_index, betas, accountant_history}
    """
    results = {'model_index' : model_index}
    print("Model index {}".format(model_index))

    df, training_data, test_data  = load_data(verbose = False)
    train_dataloader = DataLoader(training_data, batch_size = batch_size )
    test_dataloader = DataLoader(test_data, batch_size=len(test_data)) # one batch for test data 

    # make test set tensors:
    X_tw, y_t = next(iter(test_dataloader))
    X_t = X_tw[:,:-1] # feature vectors

    # instantiate then train model model:
    model = LogisticRegression(X_t.shape[1], 1)

    # saving all these models seems excessive... 
    train_losses, betas, accountant_history = train_fn(model, train_dataloader, test_dataloader, loss, verbose = False, epochs = epochs, save_state = False, metadata_path = metadata_path, noise_multiplier = noise_multiplier)

    # save model metadata
    jsonify(train_losses, Path(metadata_path, f'train_losses_model_{model_index}.json').as_posix())

    results['betas'] = betas
    results['accountant_history'] = accountant_history
    return results


# Non-private model utils
def train_model_rms(model, train_dataloader, test_dataloader, loss = loss_l2,
                          verbose = True, save_state = False, learning_rate=learning_rate,
                          momentum = momentum, epochs = epochs,
                          metadata_path = None):
    """
    Trains a private model with PRVAccountant using the given optimizer (with weights).

    Args:
        model (NN class): instantiated model
        train_dataloader (torch.DataLoader): data that the model will train on, split into batches
        test_dataloader (torch.DataLoader): data that the model will test on
        verbose (bool): if True, prints the loss and epsilon (using delta) per epoch.
        save_state (bool): if True, save the model, opacuss privacy engine and optimizer
    (rest of these args are imported from config.py)
        epochs (int): number of iterations over entire dataset
        learning_rate (float): step size multiplier for optimization
        momentum (float): paramter for how much of last step to mix into this step

    Returns:
        train_loss: list of losses of training set each epoch
        beta: final model after training
    """

    opt = torch.optim.RMSprop(
        params = model.parameters(),
        lr = learning_rate,
        momentum = momentum
    )

    # to be populated
    train_losses = []

    for epoch in range(epochs): # for each epoch
        for Xw, y_ in train_dataloader: #for each batch   

            X_ = Xw[:, :-1] # all but last/weights column
            w_ = Xw[:,-1] # only weights column 

            l = loss(model, X_, y_, weight = w_, c = c)
            # take a step:
            opt.zero_grad()
            l.backward()
            opt.step()

        # this allows us to compute privacy later with a variety of deltas if we so choose
        train_losses.append(l.item())  #

        # saving privacy state
        # https://discuss.pytorch.org/t/how-to-store-the-state-and-resume-the-state-of-the-privacyengine/138538
        if save_state:
            torch.save(privacy_engine.accountant, Path(metadata_path, f'privacy_accountant_epoch_{epoch}.pth').as_posix())
            torch.save(priv_model._module.state_dict(), Path(metadata_path, f'privacy_accountant_epoch_{epoch}.pth').as_posix())
            torch.save(priv_opt._module.state_dict(), Path(metadata_path, f'privacy_accountant_epoch_{epoch}.pth').as_posix())

        if verbose and epoch % 25 == 0:
            # now record losses on train and test: 
            print(f"Epoch {epoch}")
            print("\tTraininxg Loss {}".format(l.item()))

    beta = model.linear.weight.detach().squeeze().tolist() + model.linear.bias.detach().tolist()
    return train_losses, beta


def non_priv_kernel_fn(model_index, metadata_path, loss = loss_l2, train_fn = train_model_rms, learning_rate=learning_rate, momentum = momentum, epochs = epochs, batch_size = batch_size): 
    """ This is the kernel function to be passed to multiprocess.Pool().starmap() when running experiment 
        Given the model index, utility function, its index, privacy bound epsilon, and the metadath_path, 
        this trains a DPSGD  model to approximate beta at various epsilons measured during training
        It will save the losses in metadata_path 
        It will return a dict with the betas (the model) at the corresponding epsilon, and it's info (util_index, model_index, etc. )

    Args:
        model_index (int): index of the model being trained (it's initalized in this function)
        metadata_path (str): path to results subfolder for where to store the losses during training
        ---
        training_fn (function, optional): torch training function, see train_dpsgd_model_rms() above
        loss (function, option): a loss function, see loss_fu above
    (rest of these args are training parameters, imported from config.py)    
        learning_rate (float): step size multiplier for optimization
        momentum (float): paramter for how much of last step to mix into this step
        epochs (int, optional): numer of epochs to use in training. Defaults to epochs as defined/imported in config.py.
        batch_size (int, optional): number of samples used in each batch Defaults to batch_size  as defined/imported in config.py.
        
    Returns:
        (dict): results dict {model_index, betas}
    """
    results = {'model_index' : model_index}
    print("Model index {}".format(model_index))

    df, training_data, test_data  = load_data(verbose = False)
    train_dataloader = DataLoader(training_data, batch_size = batch_size)
    test_dataloader = DataLoader(test_data, batch_size=len(test_data)) # one batch for test data 

    # make test set tensors:
    X_tw, y_t = next(iter(test_dataloader))
    X_t = X_tw[:,:-1] # feature vectors

    # instantiate then train model model:
    model = LogisticRegression(X_t.shape[1], 1)

    # saving all these models seems excessive... 
    train_losses, beta = train_fn(model, train_dataloader, test_dataloader, loss, verbose = True, epochs = epochs, save_state = False, metadata_path = metadata_path, learning_rate = learning_rate, momentum = momentum)

    # save model metadata
    jsonify(train_losses, Path(metadata_path, f'train_losses_model_{model_index}.json').as_posix())

    results['beta'] = beta
    return results


