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
from typing import Tuple, List, Dict
import numpy as np
from numpy.typing import NDArray
from torch import nn, from_numpy, Tensor, manual_seed, no_grad
import matplotlib.pyplot as plt
from torch.optim import SGD, Optimizer
from sklearn.model_selection import train_test_split

np.random.seed(0)


# **Generate data:**

def grid(scale_x: Tuple[int, int], scale_y: Tuple[int, int]) -> Tuple[NDArray, NDArray]:
    x = np.linspace(*scale_x, 30)
    y = np.linspace(*scale_y, 30)
    xx, yy = np.meshgrid(x, y)
    return xx, yy


def trigo_function(x1, x2):
    return np.sin(x1) * np.cos(x2) + 0.1 * np.random.rand(x1.shape[0], x1.shape[1])


def generate_data() -> Tuple[Tuple, NDArray]:
    plane = grid((-5, 5), (-5, 5))
    fx = trigo_function(*plane)
    return plane, fx


def vectorize_data(x: NDArray, y: NDArray, z: NDArray) -> Tuple[NDArray, NDArray]:
    xx = x.reshape(-1, 1)
    yy = y.reshape(-1, 1)
    input_data = np.hstack((xx, yy))
    return input_data, z.reshape(-1, 1)


def viz_data(x: NDArray, y: NDArray, z: NDArray, show: bool = True):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot_surface(x, y, z, color='b', alpha=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(60, 35)
    if show:
        plt.show()
    return ax


def viz_preds(data, preds):
    x, fx = vectorize_data(*data[0], data[1])
    x_train, x_test, y_train, y_test = train_test_split(x, fx, test_size=0.3, random_state=1)
    ax = viz_data(*data[0], data[1], show=False)
    ax.scatter(x_test[:, 0], x_test[:, 1], preds, color='r')
    plt.show()


def convert_to_tensor(array: NDArray):
    return from_numpy(array.astype(np.float32))


# **Define the Model:**


class RegressionModel(nn.Module):
    def __init__(self, num_inputs: int, num_neurons: List):
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


def train_model(
        model: nn.Module,
        optimizer: Optimizer,
        loss_fn: nn.MSELoss,
        x_train: Tensor, y_train: Tensor
) -> Tensor:
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = loss_fn(y_pred, y_train)
    loss.backward()
    optimizer.step()
    return loss


def test_model(model: nn.Module, x_test: Tensor, y_test: Tensor, loss_fn: nn.MSELoss) -> Tuple[Tensor, Tensor]:
    model.eval()
    with no_grad():
        y_pred = model(x_test)
        loss = loss_fn(y_pred, y_test)
    return loss, y_pred.ravel()


def run_model(model: nn.Module, data: Tuple[Tuple, NDArray], num_of_epochs: int) -> [List, List, Dict]:
    lr = 0.1
    momentum = 0.9
    train_losses = []
    test_losses = []
    loss_function = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=lr, momentum=momentum)
    x, fx = vectorize_data(*data[0], data[1])
    x_train, x_test, y_train, y_test = (convert_to_tensor(v) for v in
                                        train_test_split(x, fx, test_size=0.3, random_state=1))
    pred = np.array([1])
    for epoch in range(num_of_epochs):
        train_loss = train_model(model, optimizer, loss_function, x_train, y_train)
        test_loss, test_pred = test_model(model, x_test, y_test, loss_function)
        train_losses.append(train_loss.item())
        test_losses.append(test_loss.item())
        pred = test_pred.detach().numpy()
        if (epoch + 1) % 100 == 0:
            print('epoch:', epoch + 1, ',train_loss =', train_loss.item())
            print('epoch:', epoch + 1, ',test_loss =', test_loss.item())
        # validation usage
    return train_losses, test_losses, dict(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test, pred=pred)


# **Visualizing the plots:**


kw = {'title': 'Epochs Vs. MSE', 'x_label': 'Epochs', 'y_label': 'MSE'}


def viz_epochs(num_of_epochs: int,
               other_axis: list,
               plot_test: bool,
               # title: str,
               # x_label: str,
               # y_label: str,
               y_lim: tuple = (0.22, 0.28)
               ):
    epochs = list(range(num_of_epochs))
    plt.plot(epochs, other_axis[0], 'orange', label='Train MSEs')
    if plot_test:
        plt.plot(epochs, other_axis[1], 'blue', label='Test MSEs', linestyle='--')
    # plt.title(title)
    # plt.xlabel(x_label)
    # plt.ylabel(y_label)
    plt.ylim(list(y_lim))
    plt.legend()
    plt.show()


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

# **Training and validation:**

# **Visualizing the plots:**

"""
5. Briefly explain graph's results.
6. How does your metric value differs between the training data and the test data and why?
"""


def run_script(regression_model: nn.Module, epochs: int):
    manual_seed(1202)
    data = generate_data()
    viz_data(*data[0], data[1])
    train_losses, test_losses, splited_data = run_model(model=regression_model, data=data, num_of_epochs=epochs)
    viz_epochs(num_of_epochs=epochs, other_axis=[train_losses, test_losses], plot_test=True, )
    viz_preds(data, splited_data['pred'], )


if __name__ == '__main__':
    model1 = RegressionModel(2, [3, 3])
    model2 = OverfitModel(2, [5, 5, 5])
    run_script(model1, 1000)
    run_script(model2, 10000)
