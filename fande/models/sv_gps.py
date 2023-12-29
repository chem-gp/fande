import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood


from gpytorch.variational import VariationalStrategy
from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution, DeltaVariationalDistribution

import numpy as np

from pytorch_lightning import LightningModule


class SVGPModel(ApproximateGP):
        def __init__(self, inducing_points):

            var_distribution = "cholesky"
            learn_inducing_points = True

            if var_distribution=="cholesky":
                    variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            elif var_distribution=="mean_field":
                    variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(-2))
            elif var_distribution=="delta":
                    variational_distribution = DeltaVariationalDistribution(inducing_points.size(0))
            
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_points)
            # variational_strategy = BatchDecoupledVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True, mean_var_batch_dim=-1)

            super().__init__(variational_strategy)
            # self.mean_module = gpytorch.means.ConstantMean()
            self.mean_module = gpytorch.means.ZeroMean()
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=720))
            soap_dim = inducing_points.size(-1)
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=soap_dim))
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=soap_dim))
            # self.covar_module = gpytorch.kernels.LinearKernel()

        def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
