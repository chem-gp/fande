import torch
import gpytorch

from gpytorch.kernels import RBFKernel, ScaleKernel, LinearKernel, AdditiveKernel, MaternKernel
from gpytorch.means import ZeroMean, ConstantMean

from gpytorch.distributions import MultitaskMultivariateNormal

from gpytorch.lazy import MatmulLazyTensor

from .my_kernels import CustomKernel

from gpytorch.models import ExactGP

from pytorch_lightning import LightningModule, Trainer, seed_everything

from torch.utils.data import DataLoader, TensorDataset, random_split

try:
    import wandb
except ImportError:
    print("wandb not installed, skipping import")


from torch.optim import Adam


import numpy as np

import fande



from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy

class SVGPModelForces(ApproximateGP):
    """
    Docs for SVGP model of forces...
    """
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
        self, train_X, train_Y, likelihood, soap_dim=None
    ): 
        super().__init__(train_X, train_Y, likelihood) # the old-style super(ExactGPModel, self) was causing error!
        
        print(self.hparams)

        if train_X is not None:
            self.soap_dim = train_X.shape[-1]
        else:
            print(self.hparams)
            self.soap_dim =soap_dim

        # self.mean_module = ZeroMean()
        self.covar_module = ScaleKernel( MaternKernel(ard_num_dims=self.soap_dim) )#LinearKernel()
        # self.covar_module = ScaleKernel( RBFKernel() )
        # self.covar_module = LinearKernel()
        # self.mean_module = ConstantMean()
        self.mean_module = ZeroMean()
        # self.covar_module = MaternKernel(ard_num_dims=self.soap_dim)#LinearKernel()
        # self.covar_module = ScaleKernel( MaternKernel(ard_num_dims=self.soap_dim) )#LinearKernel()

        # self.covar_module = LinearKernel()

        # if hparams is not None:
        #     self.hparams.update(hparams)


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
    r"""
    Batch independent multioutput exact gp model.

    .. math::
        \begin{equation}
        k_{i}(\mathbf{x}, \mathbf{x'}) 
        \end{equation}

    .. note::
        Simple note.
    """

    def __init__(
            self, 
            train_x=None, 
            train_y=None,
            atomic_group=None, 
            hparams=None,
            id=0):
        """Initialize gp model with mean and covar."""
        super().__init__()

        # self.hparams = hparams

        if hparams is not None:
            self.hparams.update(hparams)
        
        # self.save_hyperparameters()
        #add prior for badly conditioned datasets
        # this prior can affect predictions badly
        # see https://github.com/cornellius-gp/gpytorch/issues/1297
    #     self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
    # noise_prior=gpytorch.priors.SmoothedBoxPrior(0.1, 1.5, sigma=0.001))

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        if train_x is not None:
            self.soap_dim = train_x.shape[-1]  
        # soap_dim = train_x.shape[-1]    
        self.model = ExactGPModelForces(train_x, train_y, self.likelihood, soap_dim=self.soap_dim)       
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)
        
        self.atomic_group = atomic_group

        ## Store the training parameters inside the model:
        self.train_x = train_x
        self.train_y = train_y

        self.id = id

        # self.num_epochs = 10
        self.learning_rate = self.hparams['per_model_hparams'][id]['learning_rate']
        # self.precision = 32 

       
        self.save_hyperparameters(ignore=['train_x', 'train_y'])


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
        wandb.log({f"train/loss_{self.id}": loss})
        # self.log("loss", loss, prog_bar=True, on_step=False, on_epoch=True) # unfortunately slows down the training

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
        """wrapper around lightning function"""
        return self(batch[0])
    
    def predict_forces(self,input_):
        """
        Just prediction step with additional reshape taking into account the atomic group

        Returns:
            reshaped prediction
        """
        return self(input_).mean.reshape(-1, len(self.atomic_group), 3)
    
    # def info(self, dictionary: dict) -> None:
    #     for key, value in dictionary.items():
    #         self.log(key, value, prog_bar=True)

    # def on_training_epoch_end(self, outputs) -> None:
        # loss = sum(output['loss'] for output in outputs) / len(outputs)
        # print(loss)




## class that includes the collection of ModelForces models

class GroupModelForces(LightningModule):
    """
    Class that includes the collection of ModelForces models.
    """

    def __init__(
            self,
            models: list,
            train_data_loaders: list,
            fdm=None, # specification of fdm is optional
            hparams=None,
                 ) -> None:
        super().__init__()

        self.models = models
        self.train_data_loaders = train_data_loaders

        self.trainers = []
        self.per_model_hparams = hparams['per_model_hparams']

        for idx, model in enumerate(self.models):
            trainer = Trainer(accelerator='gpu', devices=1, max_epochs=self.per_model_hparams[model.id]['num_epochs'], precision=32)
            self.trainers.append(trainer)

        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.fdm = fdm

         #, mode="disabled")




    def forward(self, x):
        """Compute prediction."""

        res = []
        for model in self.models:
            res.append(model(x))

        return res
    
    def fit(self):
        """
        Train all force models associated with atomic groups. Now it is done sequentially.
        """

        fande.logger.info("Training force models")

        for idx, model in enumerate(self.models):
            print(f"Training force model {idx} of {len(self.models)}")
            self.trainers[idx].fit(model, self.train_data_loaders[idx]) #, log_every_n_steps=10)



    def eval(self):

        for model in self.models:
            model.eval()

        return
    



