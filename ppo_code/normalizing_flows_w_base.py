import torch.nn as nn
import torch

# Function to sample from base distribution
def random_normal_samples(n, dim=1):
    return torch.zeros(n, dim).normal_(mean=0, std=1)

from torch.distributions.multivariate_normal import MultivariateNormal as MV


class BaseNormal(nn.Module): 
        
    def __init__(self, D):
        super(BaseNormal, self).__init__()
        self.dim = D
        self.mu = nn.Parameter(torch.ones(1,D))
        self.var = nn.Parameter(torch.eye(2))

    def forward(self):
        return MV(loc = self.mu, covariance_matrix= self.var).sample()


class PlanarFlow(nn.Module): # per-layer class for a single Planar Flow layer

    """
    A single planar flow, 
    - computes `T(z) = z + h( <z,w> + b)u` where parameters `w,u` are vectors, `b` is a scalar, `h` is tanh activation function. 
    - log(det(|jacobian_T|)))
    """
    def __init__(self, D):
        super(PlanarFlow, self).__init__()
        self.u = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.w = nn.Parameter(torch.Tensor(1, D), requires_grad=True)
        self.b = nn.Parameter(torch.Tensor(1), requires_grad=True)
        self.h = torch.tanh
        self.dim = D
        self.init_params()

    def init_params(self):
        self.w.data.uniform_(-0.01, 0.01)
        self.b.data.uniform_(-0.01, 0.01)
        self.u.data.uniform_(-0.01, 0.01)

    def forward(self, z):
        linear_term = torch.mm(z, self.w.T) + self.b
        return z + self.u * self.h(linear_term)

    def h_prime(self, x):
        """
        Derivative of tanh
        """
        return 1 - (self.h(x)).pow(2)

    def psi(self, z):
        # auxiliary vector computed for the log_det_jacobian
        # f'(z) = I + u \psi^t, so det|f'| = 1 + u^t \psi
        inner = torch.mm(z, self.w.T) + self.b
        return self.h_prime(inner) * self.w

    def log_det_jac(self, z):
        inner = 1 + torch.mm(self.psi(z), self.u.T)
        return torch.log(torch.abs(inner))




class NormalizingFlowWithBase(nn.Module):
    """
    A normalizng flow composed of a sequence of planar flows.
    superclass is torch.nn
    """
    def __init__(self, D, n_flows=2):
        """Initiates a NF class with a sequence of planar flows. 
            - runs the super class .init
            - creates a new attribute, `self.flows` that is ModuleList (a PyTorch object for having lists of PyTorch objects) of the planar flows in each layer.

        Args:
            D (int): dimension of this flow
            n_flows (int, optional): How many layers of the NF, defaults to 2.
        
        
        """
        super(NormalizingFlowWithBase, self).__init__()
        self.flows = nn.ModuleList(
            [BaseNormal(D)] +\
            [PlanarFlow(D) for _ in range(n_flows)])
        self.dim = D

    def sample(self, n):
        """
        Transform samples from a simple base distribution
        by passing them through a sequence of Planar flows.

        Args:
            n (int): number of samples

        Returns:
            samples (torch.tensor): transformed samples 
        """
        BN = self.flows[0]
        samples = torch.vstack([BN.forward() for _ in range(n)])
        for flow in self.flows[1:]:
            samples = flow(samples)
        return samples

    def forward(self, n):
        """
        Args:
            n (int): number of samples

        Computes and returns: 
        -  T(x) = f_k\circ ... \circ f_1(x) (the transformed samples)
        - \log( |\det T'(x)|) = sum \log[ \det |f_i'(x_i)|]
        """
        sum_log_det = 0

        BN = self.flows[0] # base normal distribution, each sample is dimension dim
        transformed_sample = torch.vstack([ BN.forward() for _ in range(n)]) # n by dim
        
        for i in range( 1, len(self.flows) ):
            log_det_i = (self.flows[i].log_det_jac(transformed_sample))
            sum_log_det += log_det_i
            transformed_sample = self.flows[i](transformed_sample)

        return transformed_sample, sum_log_det
    
