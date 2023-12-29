import fande

import torch
import gpytorch

from pytorch_lightning import LightningModule, Trainer, seed_everything

from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR

from .sv_gps import SVGPModel
from .exact_gps import ExactGPModel


try:
    import wandb
except ImportError:
    print("wandb not installed, skipping import")

class ModelForces(LightningModule):
    r"""
    ...
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

        self.atomic_group = atomic_group
        ## Store the training parameters inside the model:
        self.train_x = train_x
        self.train_y = train_y
        self.id = id

        if self.hparams['per_model_hparams'][id]['forces_model_hparams']['model_type'] == "exact":
            self.model = ExactGPModel(self.train_x, self.train_y, self.likelihood, soap_dim=self.soap_dim)    
            # self.model =  DKLModelForces(train_x, train_y, self.likelihood, soap_dim=self.soap_dim)
            self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(
                self.likelihood, self.model)
        elif self.hparams['per_model_hparams'][id]['forces_model_hparams']['model_type'] == "variational_inducing_points":
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
        self.log("loss", loss, prog_bar=True) #
        return {'loss': loss}

    def on_train_step_end(self, outputs) -> None:
        # loss = sum(output['loss'] for output in outputs) / len(outputs)
        print("loss output...")

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

        seed_everything(42, workers=True)

        self.models = models
        self.train_data_loaders = train_data_loaders

        if gpu_id is not None:
            self.gpu_id = gpu_id

        self.trainers = []
        self.per_model_hparams = hparams['per_model_hparams']

        for idx, model in enumerate(self.models):
            if torch.cuda.is_available():
                trainer = Trainer(
                    accelerator='gpu',
                    # devices=1, 
                    devices=[self.gpu_id], 
                    max_epochs=self.per_model_hparams[model.id]['num_epochs'], 
                    precision=32,
                    deterministic=True
                    )
            else:
                trainer = Trainer(
                    accelerator='cpu',
                    # devices=1, 
                    max_epochs=self.per_model_hparams[model.id]['num_epochs'], 
                    precision=32,
                    deterministic=True
                    )
            
            self.trainers.append(trainer)

        self.hparams.update(hparams)
        self.save_hyperparameters()

        self.fdm = fdm


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
    
