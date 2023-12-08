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

from gpytorch.lazy import MatmulLazyTensor

from .my_kernels import CustomKernel

from gpytorch.models import ExactGP

from pytorch_lightning import LightningModule, Trainer, seed_everything

from torch.optim import Adam


from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    MeanFieldVariationalDistribution,
    DeltaVariationalDistribution,
)
from gpytorch.variational import VariationalStrategy


from .exact_gps import ExactGPModelEnergies

# from .sv_gps import SVGPModelEnergies

# from .deep_gps import DeepGP_Model

from gpytorch.mlls import DeepApproximateMLL

import fande



class ExactGPModelEnergy(ExactGP, LightningModule):
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




class RawEnergyModel(LightningModule):
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
            hparams=None):
        """
        Initialize gp model with mean and covar.
        """
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

        ## Store the training parameters inside the model:
        self.train_x = train_x
        self.train_y = train_y

            
        self.model = ExactGPModelEnergy(self.train_x, self.train_y, self.likelihood, soap_dim=self.soap_dim)    
        # self.model =  DKLModelForces(train_x, train_y, self.likelihood, soap_dim=self.soap_dim)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
            self.likelihood, self.model)
        
        # self.atomic_group = atomic_group



        # self.id = id

        # self.num_epochs = 10
        self.learning_rate = self.hparams['energy_model_hparams']['learning_rate']
        # self.precision = 32 

       
        self.save_hyperparameters(ignore=['train_x', 'train_y'])

        print("RawEnergyModel initialized")


   
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




class EnergyModel(LightningModule):
    """
    Class that includes the collection of ModelForces models.
    """

    def __init__(
            self,
            # energy_model,
            energy_train_data_loader,
            fdm=None, # specification of fdm is optional
            hparams=None,
            gpu_id=None
                 ) -> None:
        super().__init__()

        self.train_data_loader = energy_train_data_loader

        raw_energy_model = RawEnergyModel(
            train_x = energy_train_data_loader.dataset[:][0],
            train_y = energy_train_data_loader.dataset[:][1],
            hparams = hparams)

        self.model = raw_energy_model


        if gpu_id is not None:
            self.gpu_id = gpu_id
        else:
            self.gpu_id = 0

        self.trainer = None 
        self.energy_model_hparams = hparams['energy_model_hparams']

        trainer = Trainer(
                accelerator='gpu',
                # devices=1, 
                devices=[self.gpu_id], 
                max_epochs=self.energy_model_hparams['num_epochs'], 
                precision=32
                )

        self.trainer = trainer
        self.hparams.update(hparams)
        self.save_hyperparameters()
        self.fdm = fdm


    def forward(self, x):
        """Compute prediction."""
        res = self.model(x)
        return res
    

    def fit(self):
        """
        Train energy model.
        """
        fande.logger.info("Training energy model.")
        print(f"Training energy model")
        self.trainer.fit(self.model, self.train_data_loader)
        return


    def eval(self):
        self.model.eval()
        return