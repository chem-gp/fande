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

# from pytorch_lightning.plugins.environments import ClusterEnvironment

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






############################### DKL model #########################

class LargeFeatureExtractor(torch.nn.Sequential):
    def __init__(self, data_dim, output_dim):
        super(LargeFeatureExtractor, self).__init__()
        #   self.add_module('batchnorm0', torch.nn.BatchNorm1d(data_dim))

        self.add_module("linear1", torch.nn.Linear(data_dim, 1000))
        # torch.nn.init.normal_(self.linear1.weight, mean=0, std=1.0) # this init doesn't work!
        #   self.add_module('batchnorm1', torch.nn.BatchNorm1d(1000))

        self.add_module("relu1", torch.nn.ReLU())

        self.add_module("linear2", torch.nn.Linear(1000, 50))
        # torch.nn.init.normal_(self.linear2.weight, mean=0, std=1.0)
        #   self.add_module('batchnorm2', torch.nn.BatchNorm1d(500))
        self.add_module("relu2", torch.nn.ReLU())

        self.add_module("linear3", torch.nn.Linear(50, output_dim))
        # torch.nn.init.normal_(self.linear3.weight, mean=0, std=1.0)
        #   self.add_module('batchnorm3', torch.nn.BatchNorm1d(10))
        #   self.add_module('relu3', torch.nn.ReLU())

        #   self.add_module('batchnorm3', torch.nn.BatchNorm1d(100))
        #   self.add_module('linear4', torch.nn.Linear(10, 2))
        #   self.add_module('relu4', torch.nn.ReLU())

        #   self.add_module('linear5', torch.nn.Linear(10, output_dim))
        #   self.add_module('batchnorm3', torch.nn.BatchNorm1d(10))
        self.add_module("tanh5", torch.nn.Tanh())
    #   self.add_module('softmax5', torch.nn.Softmax()) # produces strange 1D line
    #   self.add_module('sigmoid5', torch.nn.Sigmoid())


class DKLModelForces(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, soap_dim):

        super(DKLModelForces, self).__init__(train_x, train_y, likelihood)

        # self.hparams = hparams

        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = gpytorch.means.ZeroMean()

        dkl_output_dim = 3
        grid_size = 10

        feature_extractor = LargeFeatureExtractor(
            data_dim=soap_dim, output_dim=dkl_output_dim
        )

        # self.covar_module = gpytorch.kernels.GridInterpolationKernel(
        #     gpytorch.kernels.ScaleKernel(
        #         gpytorch.kernels.MaternKernel(
        #             ard_num_dims=dkl_output_dim, nu=0.5
        #         )
        #     ),
        #     num_dims=dkl_output_dim,
        #     grid_size=grid_size,
        # )          

        self.covar_module = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.MaternKernel(
                    ard_num_dims=dkl_output_dim, nu=0.5
                )
            )      

        # self.covar_module = gpytorch.kernels.ScaleKernel(
        #     gpytorch.kernels.MaternKernel(
        #         ard_num_dims=hparams.gp_dkl_ard_dim, nu=0.5
        #     )
        # )

        self.feature_extractor = feature_extractor
            # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

    def forward(self, x):
        # We're first putting our data through a deep net (feature extractor)
        projected_x = self.feature_extractor(x)
        projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

################################################################################







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
        # self.model =  DKLModelForces(train_x, train_y, self.likelihood, soap_dim=self.soap_dim)
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

        print("ModelForces initialized")


   
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
        # wandb.log({f"loss_{self.id}": loss})
        # wandb.log({"loss": loss})
        # self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True) # unfortunately slows down the training

        return {'loss': loss}

    def on_train_step_end(self, outputs) -> None:
        # loss = sum(output['loss'] for output in outputs) / len(outputs)
        print("loss output...")

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
            gpu_id=None
                 ) -> None:
        super().__init__()

        self.models = models
        self.train_data_loaders = train_data_loaders

        if gpu_id is not None:
            self.gpu_id = gpu_id

        self.trainers = []
        self.per_model_hparams = hparams['per_model_hparams']

        for idx, model in enumerate(self.models):
            trainer = Trainer(
                accelerator='gpu',
                # devices=1, 
                devices=[self.gpu_id], 
                max_epochs=self.per_model_hparams[model.id]['num_epochs'], 
                precision=32
                )
            self.trainers.append(trainer)

        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.fdm = fdm

         #, mode="disabled")

# https://lightning.ai/docs/pytorch/stable/common/trainer.html


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

        # import joblib
        # def train_single_model(idx):
        #     print(f"Training force model {idx} (Total {len(self.models)} models)")
        #     self.trainers[idx].fit(self.models[idx], self.train_data_loaders[idx])
        #     return
        # joblib.Parallel(n_jobs=len(self.models),prefer="threads")(joblib.delayed(train_single_model)(idx) for idx in range(len(self.models)))

        for idx, model in enumerate(self.models):
            print(f"Training force model {idx} (Total {len(self.models)} models)")
            self.trainers[idx].fit(model, self.train_data_loaders[idx])


    def eval(self):
        for model in self.models:
            model.eval()
        return
    







