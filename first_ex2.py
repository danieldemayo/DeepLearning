# Regression with Neural Networks

"""
In this part of the exercise you will need to implement a regression model using neural networks.
The model should predict the output of a trigonometric function of two variables.
Your data set is based on a meshgrid.
Your task is to create a list of points that would correspond to a grid and use it for the input of your neural network.
Then, build your neural networks and find the architecture which gives you the best results.
1. Plot the surface from the overall data and compare it to your predicted test sets.
2. Which loss function and validation metric did you choose?
3. Plot the loss and validation metrics vs epoch for the training and test sets.
4. Build a new neural network and try overfitting your training set. Show the overfitting by using learning curve plots.
    **Note**: You can use plt.ylim() function to better focus on the changes in the trends.
"""

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, from_numpy, Tensor
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.optim import SGD
from sklearn.model_selection import train_test_split

np.random.seed(0)


# **Generate data:**

def grid(scale_x: tuple([int, int]), scale_y: tuple([int, int])) -> tuple([NDArray, NDArray]):
    x = np.linspace(*scale_x, 30)
    y = np.linspace(*scale_y, 30)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def trigo_function(x1, x2):
    return np.sin(x1) * np.cos(x2) + 0.1 * np.random.rand(x1.shape[0], x1.shape[1])


def generate_data() -> tuple([NDArray, NDArray, NDArray]):
    xx, yy = grid((-5, 5), (-5, 5))
    z = trigo_function(xx, yy)
    return xx, yy, z


def vectorize_data(x, y, z):
    xx = x.ravel().reshape(-1, 1)
    yy = y.ravel().reshape(-1, 1)
    input_data = np.hstack((xx, yy))
    return input_data, z.reshape(-1, 1)


def viz_data(x: NDArray, y: NDArray, z: NDArray):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(x, y, z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(60, 35)
    plt.show()


def convert_to_tensor(array: NDArray):
    return from_numpy(array.astype(np.float32))


# **Define the Model:**


class RegressionModel(nn.Module):
    def __init__(self, num_inputs: int, num_neurons: list):
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, num_neurons[0])
        self.lin2 = nn.Linear(num_neurons[0], num_neurons[1])
        self.lin3 = nn.Linear(num_neurons[1], 1)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x: Tensor):
        x = self.relu(self.lin1(x))
        x = self.sig(self.lin2(x))
        x = self.sig(self.lin3(x))
        return x


# **Training and validation:**
model_conf = {
    'loss_function': nn.MSELoss,
    'optimizer': SGD,
    'lr': 0.1,
    'momentum': 0.9,
    'num_of_epochs': 1000,
}


def train_model(model: nn.Module, conf: dict, x_train: Tensor, y_train: Tensor, x_test: Tensor, y_test: Tensor) -> tuple([nn.Module, list]):
    losses = []
    predictions = []
    mses = []
    r2s = []
    test_mses = []
    test_r2 = []
    criterion = conf['loss_function']()
    optimizer = conf['optimizer'](model.parameters(), lr=conf['lr'], momentum=conf['momentum'])

    for epoch in range(conf['num_of_epochs']):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        predictions.append(y_pred)
        mse = mean_squared_error(y_train.detach().numpy(), y_pred.detach().numpy())
        mses.append(mse)
        r2_scores = mean_absolute_error(y_train.detach().numpy(), y_pred.detach().numpy()) #r2_score
        r2s.append(r2_scores)
        if (epoch + 1) % 100 == 0:
            print('epoch:', epoch + 1, ',loss=', loss.item())

        _, test_mse, r2 = test_model(model=model, x_test=x_test, y_test=y_test, loss_fn=criterion)
        test_mses.append(test_mse)
        test_r2.append(r2)

    scores = [losses, predictions, mses, test_mses, r2s,test_r2]
    return model, scores


def test_model(model: nn.Module, x_test: Tensor, y_test: Tensor, loss_fn: nn.MSELoss) -> tuple([nn.Module, list]):
    model.eval()

    with torch.no_grad():
        y_pred = model(x_test)
        loss = loss_fn(y_pred, y_test)
    test_preds = y_pred.detach().numpy()
    r2 = mean_absolute_error(y_test.detach().numpy(), test_preds)

    return model, loss, r2


def validate_model(model: nn.Module, x_test, y_test):
    y_pred = model(x_test)
    mse = mean_squared_error(y_test.detach().numpy(), y_pred.detach().numpy())
    print(mse)
    return mse


# **Visualizing the plots:**


kw = {'title': 'Epochs Vs. MSE', 'x_label': 'Epochs', 'y_label': 'MSE'}


def viz_epochs(num_of_epochs: int, other_axis: list, plot_test: bool, title: str, x_label: str, y_label: str):
    epochs = list(range(num_of_epochs))
    plt.plot(epochs, other_axis[0], 'orange', label='Train MSEs')
    if plot_test:
        plt.plot(epochs, other_axis[1], 'blue', label='Test MSEs', linestyle='--')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.ylim([0.22, 0.28])
    plt.legend()
    plt.show()


# def viz_epochs2(num_of_epochs: int, other_axis: list, title: str, x_label: str, y_label: str, ):
#     epochs = list(range(num_of_epochs))
#     plt.plot(epochs, other_axis)
#     plt.plot(epochs, other_axis)
#     plt.title(title)
#     plt.xlabel(x_label)
#     plt.ylabel(y_label)
#     plt.legend()
#     plt.show()


# Build a new neural network and try overfitting your training set

# **Generate data:**

# **Define the Model:**

class OverfitModel(nn.Module):
    def __init__(self, num_inputs: int, num_neurons: list):
        super().__init__()
        self.lin1 = nn.Linear(num_inputs, num_neurons[0])
        self.lin2 = nn.Linear(num_neurons[0], num_neurons[1])
        self.lin3 = nn.Linear(num_neurons[1], num_neurons[2])
        self.lin4 = nn.Linear(num_neurons[2], 1)

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.lin1(x))
        x = self.sig(self.lin2(x))
        x = self.sig(self.lin3(x))
        x = self.sig(self.lin4(x))
        return x


# **Training and validation:**
model_conf_overfit = {
    'loss_function': nn.MSELoss,
    'optimizer': SGD,
    'lr': 0.1,
    'momentum': 0.9,
    'num_of_epochs': 10000,
}

# **Training and validation:**

# **Visualizing the plots:**

"""
5. Briefly explain graph's results.
6. How does your metric value differs between the training data and the test data and why?
"""


def main():
    torch.manual_seed(1202)
    data = generate_data()
    viz_data(*data)
    v_data = vectorize_data(*data)
    reg_model = RegressionModel(2, [3, 3])
    X_train, X_test, y_train, y_test = [convert_to_tensor(data_set) for data_set in
                                        train_test_split(*v_data, test_size=0.3)]
    trained_model, tr_scores = train_model(reg_model, model_conf, x_train=X_train, y_train=y_train,x_test=X_test,y_test=y_test)

    viz_epochs(model_conf['num_of_epochs'], [tr_scores[2], tr_scores[3]], plot_test=True, **kw)
    viz_epochs(model_conf['num_of_epochs'], [tr_scores[4], tr_scores[5]], plot_test=True, **kw)
    # y_pred = trained_model(X_test)

    # viz_data(X_test[:, 0], X_test[:, 1], y_test)
    # viz_data(X_test[:, 0], X_test[:, 1], y_pred)

    ## overfit part
    reg_overfit_model = OverfitModel(2, [5, 5, 5])
    trained_model_overfit, tr_scores_overfit = train_model(reg_overfit_model, model_conf_overfit, x_train=X_train, y_train=y_train, x_test=X_test,
                                           y_test=y_test)

    viz_epochs(model_conf_overfit['num_of_epochs'], [tr_scores_overfit[2], tr_scores_overfit[3]], plot_test=True, **kw)


if __name__ == '__main__':
    main()
