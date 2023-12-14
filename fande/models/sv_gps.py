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


from gpytorch.variational import CholeskyVariationalDistribution, MeanFieldVariationalDistribution

import numpy as np


# class SVGPModelEnergies(ApproximateGP):
#     def __init__(self, inducing_points):

#         variational_distribution = CholeskyVariationalDistribution(
#             inducing_points.size(0)
#         )

#         # variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(0))
#         # variational_distribution = DeltaVariationalDistribution(inducing_points.size(0))

#         variational_strategy = VariationalStrategy(
#             self,
#             inducing_points,
#             variational_distribution,
#             learn_inducing_locations=True,
#         )
#         super().__init__(variational_strategy)
#         self.mean_module = gpytorch.means.ConstantMean()
#         # self.mean_module = gpytorch.means.ZeroMean()
#         # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=720))
#         # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
#         self.covar_module = gpytorch.kernels.LinearKernel()

#         # feature_extractor = LargeFeatureExtractor(720, 2)
#         # self.feature_extractor = feature_extractor
#         # self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(-1.0, 1.0)

#     def forward(self, x):
#         # projected_x = self.feature_extractor(x)
#         # projected_x = self.scale_to_bounds(projected_x)  # Make the NN values "nice"
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



class SVGPModelEnergies(ApproximateGP):
        def __init__(self, inducing_points):

            var_distribution = "cholesky"
            learn_inducing_points = True

            if var_distribution=="cholesky":
                    variational_distribution = CholeskyVariationalDistribution(inducing_points.size(0))
            elif var_distribution=="mean_field":
                    variational_distribution = MeanFieldVariationalDistribution(inducing_points.size(-2))
            elif var_distribution=="delta":
                    variational_distribution = DeltaVariationalDistribution(inducing_points.size(0))
            
            variational_strategy = VariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=learn_inducing_points)
            # variational_strategy = BatchDecoupledVariationalStrategy(self, inducing_points, variational_distribution, learn_inducing_locations=True, mean_var_batch_dim=-1)

            super().__init__(variational_strategy)
            self.mean_module = gpytorch.means.ConstantMean()
            # self.mean_module = gpytorch.means.ZeroMean()
            # self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(ard_num_dims=720))
            soap_dim = inducing_points.size(-1)
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(ard_num_dims=soap_dim))
            # self.covar_module = gpytorch.kernels.LinearKernel()

        def forward(self, x):
                mean_x = self.mean_module(x)
                covar_x = self.covar_module(x)
                return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



# To train:
# inducing_slice = np.sort( np.random.choice(np.arange(0,train_x1.size(0)), num_ind_points, replace=False) )  # replace=False - no repitions
# # print("Inducing points initializations: \n\n", inducing_slice.tolist())
# inducing_points = train_x1[inducing_slice, :]
# # inducing_points = torch.randn(100,720).cuda()


# from torch.utils.data import TensorDataset, DataLoader
# train_dataset = TensorDataset(train_x, train_y_norm)
# train_loader = DataLoader(train_dataset, batch_size=10028, shuffle=False)


# model = GPModel(inducing_points=inducing_points)
# likelihood = gpytorch.likelihoods.GaussianLikelihood()


# scheduler = MultiStepLR(optimizer, milestones=[1000, 10_000], gamma=0.1)


# if mll_loss=="var_elbo":
#         mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=train_y_norm.size(0), beta=mll_beta)
# elif mll_loss=="pred_ll":
#         mll = gpytorch.mlls.PredictiveLogLikelihood( likelihood, model, num_data=train_y_norm.size(0), beta=mll_beta ) # this is probably the most robust
# elif mll_loss=="gamma_var_elbo":
#         mll = gpytorch.mlls.GammaRobustVariationalELBO(likelihood, model, num_data=train_y_norm.size(0), beta=mll_beta, gamma=1.03)

# epochs_iter = tqdm.tqdm(range(num_epochs))
# for i in epochs_iter:
#         # Within each iteration, we will go over each minibatch of data
#         # minibatch_iter = tqdm.tqdm(train_loader, desc="Minibatch", leave=False)
#         # for x_batch, y_batch in minibatch_iter:
#         #     optimizer.zero_grad()
#         #     output = model(x_batch)
#         #     loss = -mll(output, y_batch)
#         #     # print(loss.item())
#         #     epochs_iter.set_description(f"Loss {loss.item()} ")
#         #     loss.backward()
#         #     # scheduler.step()
#         #     # print(loss.item())
#         #     optimizer.step()
#         #     # epochs_iter.set_postfix(loss=loss.item())
#         optimizer.zero_grad()
#         output = model(train_x)
#         loss = -mll(output, train_y_norm)
#         # print(loss.item())
#         epochs_iter.set_description(f"Loss {loss.item()} ")
#         wandb.log({"loss": loss.item()})
#         loss.backward()
#         optimizer.step()
#         scheduler.step()
#         # minibatch_iter.set_postfix(loss=loss.item())