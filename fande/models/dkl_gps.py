import torch
import gpytorch


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


class DKLModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, soap_dim):
        super(DKLModel, self).__init__(train_x, train_y, likelihood)
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

