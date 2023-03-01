import torch
import gpytorch

from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel, AdditiveKernel, MaternKernel
from gpytorch.means import ZeroMean, ConstantMean

from gpytorch.distributions import MultitaskMultivariateNormal

from gpytorch.lazy import MatmulLazyTensor

from .my_kernels import CustomKernel

from gpytorch.models import ExactGP

from pytorch_lightning import LightningModule

import wandb

from torch.optim import Adam



from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class SVGPModelForces(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
        variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True)
        super().__init__(variational_strategy)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()
        # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
        self.covar_module = gpytorch.kernels.LinearKernel()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class ExactGPModelForces(ExactGP, LightningModule):
    def __init__(
        self, train_X, train_Y, likelihood
    ): 
        super().__init__(train_X, train_Y, likelihood) # the old-style super(ExactGPModel, self) was causing error!
        
        self.soap_dim = train_X.shape[-1]

        # self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel( MaternKernel(ard_num_dims=self.soap_dim) )#LinearKernel()
        # self.covar_module = ScaleKernel( RBFKernel() )
        # self.covar_module = LinearKernel()
        self.mean_module = ConstantMean()
        # self.covar_module = MaternKernel(ard_num_dims=self.soap_dim)#LinearKernel()
        # self.covar_module = ScaleKernel( MaternKernel(ard_num_dims=self.soap_dim) )#LinearKernel()

        # self.covar_module = LinearKernel()


    def forward(self, x):
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



class ModelForces(LightningModule):
    """batch independent multioutput exact gp model."""

    def __init__(self, train_x, train_y, hparams, learning_rate):
        """Initialize gp model with mean and covar."""
        super().__init__()

        # self.hparams = hparams

        self.hparams.update(hparams)
        self.learning_rate = learning_rate
        # self.save_hyperparameters()
        #add prior for badly conditioned datasets
        # this prior can affect predictions badly
        # see https://github.com/cornellius-gp/gpytorch/issues/1297
    #     self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
    # noise_prior=gpytorch.priors.SmoothedBoxPrior(0.1, 1.5, sigma=0.001))

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()      
        self.model = ExactGPModelForces(train_x, train_y, self.likelihood)       
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)

        # SVGP Approximate GP model with Variational ELBO as loss function
        # self.inducing_points = train_x[0:2:2000, :]
        # self.model = SVGPModelForces(inducing_points=self.inducing_points)
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=train_y.size(0))


    def get_model_parameters(self):
        ...


    
    def forward(self, input_):
        """Compute prediction."""

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self.model(input_)           
            res = self.likelihood(output) #.mean.cpu().detach().numpy()

        return res


    def training_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        """Compute training loss."""

        input_, target = batch
        output = self.model(input_)

        loss = -self.mll(output, target)
        # wandb.log({"train/loss": loss})

        return {'loss': loss}

    # def training_step(self, batch, batch_idx):
    #     '''needs to return a loss from a single batch'''
    #     # _, loss, acc = self._get_preds_loss_accuracy(batch)
    #     train_x, train_y = batch

    #     output = self(train_x)
    #     loss = -self.mll(output, train_y) ##???

    #     # Log loss and metric
    #     # self.log('train_loss', loss)
    #     # self.log('train_accuracy', acc)
    #     return loss

    def test_step(self, batch, batch_idx):
        """Compute testing loss."""
        input_, target = batch
        output = self(input_)

        loss = -self.mll(output, target)

        return {'test_loss': loss}

    def configure_optimizers(self):
        '''defines model optimizer'''
            # Use the adam optimizer
        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    # def validation_step(self, batch, batch_idx):
    #     """Compute validation loss."""
    #     input_, target = batch
    #     output = self(input_)

    #     loss = -self.mll(output, target)

    #     return {'val_loss': loss}

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self(batch[0])





