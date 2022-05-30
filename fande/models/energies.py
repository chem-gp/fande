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


from gpytorch.lazy import MatmulLazyTensor

from .my_kernels import CustomKernel

from gpytorch.models import ExactGP

from pytorch_lightning import LightningModule

from torch.optim import Adam


from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    CholeskyVariationalDistribution,
    MeanFieldVariationalDistribution,
    DeltaVariationalDistribution,
)
from gpytorch.variational import VariationalStrategy


from .exact_gps import ExactGPModelEnergies

from .sv_gps import SVGPModelEnergies

from .deep_gps import DeepGP_Model

from gpytorch.mlls import DeepApproximateMLL


class ModelEnergies(LightningModule):
    def __init__(self, train_x, train_y, fdm, hparams, learning_rate):
        
        super().__init__()

        self.num_samples = 10 # for training deep models
        self.learning_rate = learning_rate
        self.hparams.update(hparams)

        # add prior for badly conditioned datasets
        # this prior affects badly quality of predictions
        # see https://github.com/cornellius-gp/gpytorch/issues/1297
        #     self.likelihood = gpytorch.likelihoods.GaussianLikelihood(
        # noise_prior=gpytorch.priors.SmoothedBoxPrior(0.1, 2.5, sigma=0.001))

        # Basic GP model:
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        self.model = ExactGPModelEnergies(train_x, train_y, self.likelihood)
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        if fdm.normalizing_const is not None:
            self.normalizing_const = fdm.normalizing_const
            self.normalizing_shift = fdm.normalizing_shift

        # SVGP Approximate GP model with Variational ELBO as loss function
        # ind_slice = np.sort(np.random.choice(np.arange(0, 1600), 50, replace=False))
        # self.inducing_points = train_x[:, :]
        # # self.inducing_points = torch.randn(100, 2)
        # self.model = SVGPModelEnergies(inducing_points=self.inducing_points)
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # # self.mll = gpytorch.mlls.PredictiveLogLikelihood(
        # #     self.likelihood, self.model, num_data=train_y.size(0)
        # # )
        # self.mll = gpytorch.mlls.VariationalELBO(
        #     self.likelihood, self.model, num_data=train_y.size(0), beta=1.0
        # )

        # GP DKL model
        # self.likelihood = gpytorch.likelihoods.GaussianLikelihood()
        # self.model = GPDKLModel(train_x, train_y, self.likelihood)
        # self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, self.model)

        # DeepGP model:
        # # print(train_x.shape)
        # self.model = DeepGP_Model(train_x, num_hidden_dims=3)
        # # if torch.cuda.is_available():
        # #     self.model = self.model.cuda()
        # self.model.to(self.device)
        # self.mll = DeepApproximateMLL( gpytorch.mlls.PredictiveLogLikelihood(self.model.likelihood, self.model, train_x.shape[-2], beta=0.1, combine_terms=True ) )
        # # mll = DeepApproximateMLL( gpytorch.mlls.GammaRobustVariationalELBO(model.likelihood, model, num_data=train_x.shape[-2], beta=0.1, gamma=1.03) )
        # # mll = DeepApproximateMLL( VariationalELBO(model.likelihood, model, train_x.shape[-2], beta=0.1) )

    def forward(self, input_):
        """Compute prediction."""

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            output = self.model(input_)
            res = self.likelihood(output) 

        return res

    def training_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        """Compute training loss."""

        input_, target = batch

        with gpytorch.settings.num_likelihood_samples(self.num_samples):
            output = self.model(input_)
            loss = -self.mll(output, target)

        return {"loss": loss}

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

        return {"test_loss": loss}

    def configure_optimizers(self):
        """defines model optimizer"""
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
