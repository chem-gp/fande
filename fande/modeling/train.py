from tqdm import tqdm
import torch
import gpytorch

import matplotlib.pyplot as plt

import wandb


def train_model(
    model,
    likelihood,
    train_X,
    train_Y,
    test_X,
    test_Y
):

    # Use the adam optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.05
    )  # Includes GaussianLikelihood parameters
    model.train()
    likelihood.train()

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    
    # train_y = train_Y[:,-1]
    # test_y = test_Y[:,-1]

    train_y = train_Y
    test_y = test_Y

    # train_y = train_Y

    training_iter = 100
    pbar = tqdm(range(training_iter))
    # pbar = range(training_iter)
    for i in pbar:
        # Zero gradients from previous iteration
        optimizer.zero_grad()
        # Output from model
        output = model(train_X)
        
        # print(output.mean.shape)
        # print(train_Dy.shape)
        # Calc loss and backprop gradients
        loss = -mll(output, train_y)
        loss.backward()
        # print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        #     i + 1, training_iter, loss.item(),
        #     model.covar_module.base_kernel.lengthscale.item(),
        #     model.likelihood.noise.item()
        # ))

        # print('Iter %d/%d - Loss: %.3f ' % (
        #     i + 1, training_iter, loss.item()
        # ))

        optimizer.step()
        wandb.log({"loss": loss})
        pbar.set_description(f"Loss {loss.item()} ")

    model.eval()
    likelihood.eval()
    predictions = likelihood(model(test_X)).mean.cpu().detach().numpy()
    actual_values = test_y.cpu().detach().numpy()
    plt.rcParams["figure.figsize"] = (20,3)
    plt.plot(predictions, color="blue", label="predictions", linewidth=0.3)
    plt.plot(actual_values, color="red", label="actual values", linewidth=0.3)
    plt.legend()
    # plt.ylim(0, 1.0)
    plt.show()

    # print(test_energies)
    # print(predictions)

    # print(model.get_kernel_params() )

    # check how it works on training data
    # model.eval()
    # likelihood.eval()
    # predictions = likelihood(model(train_X)).mean.cpu().detach().numpy()
    # actual_values = train_y.cpu().detach().numpy()

    # plt.plot(predictions[0:100], color="blue", label="predictions")
    # plt.plot(actual_values[0:100], color="red", label="actual values")
    # plt.legend()
    # # plt.ylim(0, 1.0)
    # # wandb.log({"energies": plt})
    # plt.show()

    # print(test_y.shape)

    return 

