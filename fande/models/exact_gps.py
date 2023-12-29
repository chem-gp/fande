import torch
import gpytorch
import numpy as np

from gpytorch.kernels import (
    RBFKernel,
    ScaleKernel,
    LinearKernel,
    AdditiveKernel,
    MultitaskKernel,
    PolynomialKernel,
    MaternKernel,
)
from gpytorch.means import ZeroMean, ConstantMean

from gpytorch.models import ExactGP

from pytorch_lightning import LightningModule


class ExactGPModel(ExactGP, LightningModule):
    def __init__(
        self, train_X, train_Y, likelihood, soap_dim=None
    ): 
        super().__init__(train_X, train_Y, likelihood) # the old-style super(ExactGPModel, self) was causing error!
        
        print(self.hparams)

        if train_X is not None:
            self.soap_dim = train_X.shape[-1]
        else:
            self.soap_dim =soap_dim

        # self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel( MaternKernel(ard_num_dims=self.soap_dim) )
        # self.covar_module = ScaleKernel( RBFKernel(ard_num_dims=self.soap_dim) )
        # self.covar_module = ScaleKernel( RBFKernel() )
        # self.covar_module = LinearKernel()
        # self.mean_module = ConstantMean()
        self.mean_module = ZeroMean()
        # self.covar_module = MaternKernel(ard_num_dims=self.soap_dim)#LinearKernel()
        # self.covar_module = ScaleKernel( MaternKernel(ard_num_dims=self.soap_dim) )#LinearKernel()


    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


