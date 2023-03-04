import torch
import math

"""
Potential functions U(x) from Rezende et al. 2015
p(z) is then proportional to exp(-U(x)).
Since we log this value later in the optimized bound,
no need to actually exp().
"""


def w_1(z):
    return torch.sin((2 * math.pi * z[:, 0]) / 4)


def w_2(z):
    return 3 * torch.exp(-.5 * ((z[:, 0] - 1) / .6) ** 2)


def sigma(x):
    return 1 / (1 + torch.exp(- x))


def w_3(z):
    return 3 * sigma((z[:, 0] - 1) / .3)


def pot_1(z):
    z_1, z_2 = z[:, 0], z[:, 1]
    norm = torch.sqrt(z_1 ** 2 + z_2 ** 2)
    outer_term_1 = .5 * ((norm - 2) / .4) ** 2
    inner_term_1 = torch.exp((-.5 * ((z_1 - 2) / .6) ** 2))
    inner_term_2 = torch.exp((-.5 * ((z_1 + 2) / .6) ** 2))
    outer_term_2 = torch.log(inner_term_1 + inner_term_2 + 1e-7)
    u = outer_term_1 - outer_term_2
    return - u


def pot_2(z):
    u = .5 * ((z[:, 1] - w_1(z)) / .4) ** 2
    return - u


def pot_3(z):
    term_1 = torch.exp(-.5 * (
        (z[:, 1] - w_1(z)) / .35) ** 2)
    term_2 = torch.exp(-.5 * (
        (z[:, 1] - w_1(z) + w_2(z)) / .35) ** 2)
    u = - torch.log(term_1 + term_2 + 1e-7)
    return - u


def pot_4(z):
    term_1 = torch.exp(-.5 * ((z[:, 1] - w_1(z)) / .4) ** 2)
    term_2 = torch.exp(-.5 * ((z[:, 1] - w_1(z) + w_3(z)) / .35) ** 2)
    u = - torch.log(term_1 + term_2)
    return - u


# below functions are not from the paper

def pot_5(z): 
    X = torch.tensor([[-1.,1.], [1.,1.]])
    y = torch.tensor([0.,1.])

    y_preds = torch.sigmoid(torch.mm(X,z.T)) 
    # compute cross entropy loss for every beta given: 
    cel_fn = torch.nn.BCELoss(reduction = 'sum' ) ## instantiates binary cross entropy loss function class, see # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    cels = torch.tensor([cel_fn(y_preds[:, j], y ) for j in range(y_preds.shape[1])]) # shape is l
    
    # compute l2 norm^2 of input
    z_1, z_2 = z[:, 0], z[:, 1]
    norm2 = z_1.pow(2) + z_2.pow(2)
    return -cels - .1* norm2

def pot_6(z): # slight variant of pot_5, makes it oblong
    X = torch.tensor([[2.,1.], [6.,1.]])
    y = torch.tensor([0.,1.])

    y_preds = torch.sigmoid(torch.mm(X,z.T)) 
    # compute cross entropy loss for every beta given: 
    cel_fn = torch.nn.BCELoss(reduction = 'sum' ) ## instantiates binary cross entropy loss function class, see # https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html
    cels = torch.tensor([cel_fn(y_preds[:, j], y ) for j in range(y_preds.shape[1])]) # shape is l
    
    # compute l2 norm^2 of input
    z_1, z_2 = z[:, 0], z[:, 1]
    norm2 = z_1.pow(2) + z_2.pow(2)
    return -cels - .1* norm2
