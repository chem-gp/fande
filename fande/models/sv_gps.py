import torch
import tqdm
import gpytorch
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultivariateNormal
from gpytorch.models import ApproximateGP, GP
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood


from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution

import numpy as np


class SVGPModelEnergies(ApproximateGP):
    def __init__(self, inducing_points):

        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        # variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(0))
        # variational_distribution = DeltaVariationalDistribution(inducing_points.size(0))

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super().__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        # self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=720))
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.LinearKernel()

        # feature_extractor = LargeFeatureExtractor(720, 2)
        # self.feature_extractor = feature_extractor
        # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x):
        # projected_x = self.feature_extractor(x)
        # projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

