import torch.nn as nn
import torch

class LinearRegressionModel(torch.nn.Module):
    """
    A linear regression model (with no bias)
    """

    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(1,1, bias = False)

    def forward(self, x):
        return self.linear(x)


# reference for basic logistic regression in Pytorch: https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
class LogisticRegression(torch.nn.Module): 
    """
    A basic logistic regression module.
    """
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs


