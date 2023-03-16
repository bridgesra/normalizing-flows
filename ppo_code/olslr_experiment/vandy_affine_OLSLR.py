# python is 0-indexed! (There's a good chance I forget this and will have off by one errors)

import pandas as pd
import os 
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import flowtorch.bijectors as bij
import flowtorch.distributions as D
from matplotlib import pyplot as plt
import math
import sklearn.datasets as datasets
import itertools



# we'll start with the hieghts and weights dataset but we can generalize later

hw_dataset = pd.read_csv(os.path.join(os.path.dirname(__file__), "data/height-weight.csv"))

height_values = hw_dataset["Height(Inches)"].values
weight_values = hw_dataset["Height(Inches)"].values

# use height to infer weight (swap to swap)
# 80-20 split for train/test, we can further split train if we need to validate
train_data = [torch.tensor(height_values[0:19999]), torch.tensor(weight_values[0:19999])]
test_data = [torch.tensor(height_values[20000:25000]), torch.tensor(weight_values[20000:25000])]


# normalizing flow stuff

# training the flow
# in our case the target dist is going to be a utility function that uses the flow
# and the base distibution
def train_rkl_flow(base_dist, target_dist, data, epochs,
                   bijector, lr = 5e-3,
                   samples = 1000):
    # initialize flow
    g = D.Flow(base_dist, bijector)
    opt = torch.optim.Adam(g.parameters(), lr)
    for idx in range(epochs):
        x = list(itertools.islice(data[0], samples*idx%len(data[0]), (samples-1)+idx*samples%len(data[0])))[0]
        y = list(itertools.islice(data[1], samples*idx%len(data[1]), (samples-1)+idx*samples%len(data[1])))[0]
        opt.zero_grad()

        z = base_dist.sample([samples]) # sample from base distribution
        # This is almost certainly not updating the base distribution... we could look into
        # pytorch's autograd to see if we can specify updating the base distribution
        loss = -rkl(g, target_dist, z, x, y)

        loss.backward()
        opt.step()
    return g

# RKL in 4.2
def rkl(g, target_dist, z, x, y):
    #TODO add links
    # flowtorch's log_prob is the first two terms our our RKL loss
    #https://github.com/facebookincubator/flowtorch/blob/1ce4b1c1ac92409905aaea49fc51de4442f5ab14/flowtorch/distributions/flow.py
    #
    return sum(-g.log_prob(z) - target_dist(g, z, x, y))

# l2 error accuracy
# pass in g.normalize if providing a torch flow model
def l2_loss(beta, data):
    x = data[0]
    y = data[1]
    return torch.mean((beta*x-y).pow(2))


# probably the most important part for us
# here g is our flow and z is the samples from our base distribution
# TODO: we need to add in epsilon and sensitivity
# we should use pytorch's data loader but for now we'll just have the data zipped
def olslr_u(g, z, x, y):
    # in flow torch rnormalize should allow autodiff so we should be tracking gradients here
    return torch.square(g.rnormalize(x) - y)

# Here's where we will probably want to play around:
bijector = bij.Affine()
# I think (emphasis on think) the gradient for these are probably being computed... need to check if
# flow torch does something so they don't or if we just need to provide them to the optimizer for
# updating
base_dist = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(1), torch.ones(1)), 1)

flow = train_rkl_flow(base_dist, olslr_u, train_data, 50, bijector)

# an optimal classifier on height weight dataset
def opt_classifier(data): return (1/np.matmul(np.transpose(data[0]), data[0])) * np.transpose(data[0]) * data[1] 

print("Optimal Classifier l2 loss on test set:")
print(l2_loss(opt_classifier(test_data), test_data).item())
print("Flow's l2 loss on test set:")
# TODO I think we want to pass in a sample which would be a beta
print(l2_loss(flow.sample(test_data[0].size()), test_data).item())

# but if we do this:
print("What I orginally thought we wanted to measure... for a flow model (instead of sampling we push through the data and we get a very good l2 loss)")
# this l2 loss is very good
print(torch.mean((flow.rnormalize(test_data[0])-test_data[1]).pow(2)).item())
    


