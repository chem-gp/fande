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
)
from gpytorch.means import ZeroMean, ConstantMean

from gpytorch.models import ExactGP

class ExactGPModelEnergies(ExactGP):
    def __init__(self, train_X, train_Y, likelihood):
        super().__init__(
            train_X, train_Y, likelihood
        )  # the old-style super(ExactGPModel, self) was causing error!
        # self.mean_module = ConstantMean()

        self.soap_dim = train_X.shape[-1]
        self.mean_module = ZeroMean()
        # self.covar_module = PolynomialKernel(power=1)
        # self.covar_module = LinearKernel()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=self.soap_dim))#LinearKernel()

    def forward(self, X):
        x = X
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    ## Below is unnecessary code...
    def get_kernel_params(self):

        vars = torch.zeros(self.num_envs)

        print(f"Likelihood noise parameter: {self.likelihood.noise}")

        for i in range(self.num_envs):
            vars[i] = self.kernels_list[i].variance
            # print(self.kernels_list[i].variance)

        # for i in range(self.num_envs):
        #     print(self.kernels_list[i].raw_variance)

        return vars

    def predict_forces(self, x, Dx):

        # da_dx = -derivatives[:, 0, :, :, :] # (100, 1, 12, 3, 720)
        # da_dx_torch = torch.tensor(da_dx)
        # test_da_dx = da_dx_torch[int(0.8*n_samples):n_samples, :]
        # train_da_dx = da_dx_torch[0:int(0.8*n_samples), :]

        # Dx = train_da_dx

        Dx = -Dx[:, 0, :, :]
        print(x.shape)
        print(Dx.shape)

        # K = model(train_x).lazy_covariance_matrix
        K = self.covar_module.forward(self.train_x, self.train_x)
        K = K.add_jitter(self.likelihood.noise)
        # Eye = gpytorch.lazy.DiagLazyTensor( torch.ones( train_x.size(1) ))
        Eye = torch.diag(torch.ones(self.train_x.size(1))).double().cuda()
        K_inv = K.inv_matmul(Eye)

        K_test = self.covar_module.forward_left(
            x, Dx, self.train_x, self.train_Dx
        ).evaluate()

        W = K_inv.matmul(self.train_y).squeeze()
        f_pred = torch.matmul(K_test, W).squeeze()

        # f_pred = f_pred.transpose(0,2)#.reshape(-1, 12)

        return f_pred