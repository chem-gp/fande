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


from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL

from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution

import numpy as np


class DeepGPHiddenLayer(DeepGPLayer):
    def __init__(
        self, 
        input_dims, 
        output_dims, 
        num_inducing=128, 
        mean_type='constant', 
        learn_inducing=True, 
        ind_points=None):

        if output_dims is None:
            inducing_points = 0.01*torch.randn(num_inducing, input_dims)
            batch_shape = torch.Size([])
        else:
            inducing_points = 0.01*torch.randn(output_dims, num_inducing, input_dims)
            batch_shape = torch.Size([output_dims])

        if ind_points is not None:
            inducing_points = ind_points


        # variational_distribution = CholeskyVariationalDistribution(
        #     num_inducing_points=num_inducing,
        #     batch_shape=batch_shape
        # )

        variational_distribution = MeanFieldVariationalDistribution(
            num_inducing_points=num_inducing,
            batch_shape=batch_shape
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=learn_inducing
        )

        super().__init__(variational_strategy, input_dims, output_dims)

        if mean_type == 'constant':
            self.mean_module = ConstantMean(batch_shape=batch_shape)
            # self.mean_module = gpytorch.means.ZeroMean(batch_shape=batch_shape)
        else:
            # self.mean_module = LinearMean(input_dims)
            # self.mean_module = gpytorch.means.MultitaskMean(
        #     [gpytorch.means.LinearMean(input_size=input_dims), 
        #     gpytorch.means.LinearMean(input_size=input_dims)
        #     ], 
        #     num_tasks=2
        # ) 
            self.mean_module = LinearMean(input_size = input_dims)
            # self.mean_module = gpytorch.means.ZeroMean()
        
        self.covar_module = ScaleKernel(
            RBFKernel(batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape, ard_num_dims=None
        )

        self.mean_type = mean_type

    def forward(self, x):          
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, gpytorch.distributions.MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(gpytorch.settings.num_likelihood_samples.value(), *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))


class DeepGP_Model(DeepGP):
    def __init__(self, train_x, num_hidden_dims=3):
        
        super().__init__()

        self.likelihood = GaussianLikelihood()

        # randomly select inducing points initialization:
        inducing_slice = np.sort( np.random.choice(np.arange(0,1600), 200, replace=False) ) 

        inducing_points_1 = train_x[inducing_slice, :]

        train_x_shape = train_x.shape

        hidden_layer = DeepGPHiddenLayer(
            input_dims=train_x_shape[-1],
            output_dims=num_hidden_dims,
            num_inducing= inducing_points_1.shape[0],
            mean_type='linear',
            ind_points = inducing_points_1,
            learn_inducing=True
        )

             
        self.hidden_layer = hidden_layer.cuda()
        inducing_points_2 = self.hidden_layer(inducing_points_1).mean.mean(0)[::2]

        last_layer = DeepGPHiddenLayer(
            input_dims=num_hidden_dims,
            output_dims=None,
            # num_inducing=100,
            num_inducing=inducing_points_2.shape[0],
            ind_points= inducing_points_2,
            mean_type='constant',
            learn_inducing = True
        )

        self.last_layer = last_layer
        

    def forward(self, inputs):
        hidden_rep = self.hidden_layer(inputs) 
        output = self.last_layer(hidden_rep)
        return output


    def predict(self, test_loader):
        with torch.no_grad():
            mus = []
            variances = []
            lls = []
            for x_batch, y_batch in test_loader:
                preds = self.likelihood(self(x_batch))
                mus.append(preds.mean)
                variances.append(preds.variance)
                lls.append(self.likelihood.log_marginal(y_batch, self(x_batch)))

        return torch.cat(mus, dim=-1), torch.cat(variances, dim=-1), torch.cat(lls, dim=-1)


