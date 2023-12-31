
import torch
import gpytorch
import numpy as np

from pytorch_lightning import LightningModule, Trainer, seed_everything

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR


from .sv_gps import SVGPModel
from .exact_gps import ExactGPModel
# from .deep_gps import DeepGP_Model


try:
    import wandb
except ImportError:
    print("wandb not installed, skipping import")

import fande

class RawEnergyModel(LightningModule):
    r"""
    ...
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

        ## Store the training parameters inside the model:
        self.train_x = train_x
        self.train_y = train_y

        if self.hparams['energy_model_hparams']['model_type'] == "exact":
            self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood, soap_dim=self.soap_dim)    
            # self.model =  DKLModelForces(train_x, train_y, self.likelihood, soap_dim=self.soap_dim)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.model)

        elif self.hparams['energy_model_hparams']['model_type'] == "variational_inducing_points":
            mll_beta = 0.1
            num_ind_points = self.hparams['energy_model_hparams']['num_inducing_points']
            # random_indices = torch.randint(low=0, high=self.train_x.shape[0], size=(num_ind_points,))
            perm = torch.randperm(self.train_x.size(0))
            random_indices = perm[:num_ind_points]
            inducing_points = self.train_x[random_indices, :]
            print("Training with inducing points: ", inducing_points.shape)
            self.model = SVGPModel(inducing_points=inducing_points)
            self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
            self.mll = gpytorch.mlls.PredictiveLogLikelihood( self.likelihood, self.model, num_data=self.train_y.size(0), beta=mll_beta )


        self.learning_rate = self.hparams['energy_model_hparams']['learning_rate']
        self.save_hyperparameters(ignore=['train_x', 'train_y'])

   
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
        self.log("loss", loss, prog_bar=True) # unfortunately slows down the training
        return {'loss': loss}

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
        # # return optimizer
        scheduler = MultiStepLR(optimizer, milestones=[1_000, 10_000], gamma=0.1)
        return [optimizer], [scheduler]


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

# https://lightning.ai/docs/pytorch/stable/common/early_stopping.html#earlystopping-callback
    # def validation_step(self, batch, batch_idx):
    #     loss = ...
    #     self.log("val_loss", loss)

class EnergyModel(LightningModule):
    """
    Class that includes the collection of ModelForces models.
    """

    def __init__(
            self,
            energy_train_data_loader,
            fdm=None, 
            hparams=None,
            gpu_id=None):
        super().__init__()

        seed_everything(42, workers=True)

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
        if torch.cuda.is_available():
            trainer = Trainer(
                    accelerator='gpu',
                    # devices=2, 
                    devices=[self.gpu_id], 
                    max_epochs=self.energy_model_hparams['num_epochs'], 
                    precision=32,
                    # strategy="deepspeed",
                    log_every_n_steps=1000,
                    deterministic=True
                    )
        else:
            trainer = Trainer(
                    accelerator='cpu',
                    # devices=2,  
                    max_epochs=self.energy_model_hparams['num_epochs'], 
                    precision=32,
                    # strategy="deepspeed",
                    log_every_n_steps=1000,
                    deterministic=True
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