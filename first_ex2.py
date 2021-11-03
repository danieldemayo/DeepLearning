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
from torch import nn, from_numpy, Tensor
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error, roc_curve, auc, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.optim import SGD
from sklearn.model_selection import train_test_split


# **Generate data:**

def generate_data(seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    np.random.seed(seed)
    x = np.linspace(-5, 5, 30)
    y = np.linspace(-5, 5, 30)
    xx, yy = np.meshgrid(x, y)
    z = np.sin(xx) * np.cos(yy) + 0.1 * np.random.rand(xx.shape[0], xx.shape[1])
    return xx, yy, z


def viz_data(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(x, y, z, 50, cmap='binary')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.view_init(60, 35)
    plt.show()


def convert_to_tensor(array: np.ndarray):
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

    def forward(self, x):
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


def train_model(model: nn.Module, conf: dict, x_train: Tensor, y_train: Tensor) -> tuple[nn.Module, list]:
    losses = []
    predictions = []
    aucs = []
    criterion = conf['loss_function']()
    optimizer = conf['optimizer'](model.parameters(), lr=conf['lr'], momentum=conf['momentum'])

    for epoch in range(conf['num_of_epochs']):
        y_pred = model(x_train)
        loss = criterion(y_pred, y_train)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        losses.append(loss.item())
        predictions.append(y_pred.detach().numpy())
        #todo: roc_auc_score
        fpr, tpr, threshold = roc_curve(y_train, predictions[epoch])
        aucs.append(auc(fpr, tpr))
        if (epoch + 1) % 100 == 0:
            print('epoch:', epoch + 1, ',loss=', loss.item())

    scores = [losses, predictions, aucs]
    return model, scores


def validate_model(model: nn.Module, x_test, y_test):
    y_pred = model(x_test)
    fpr, tpr, threshold = roc_curve(y_test.detach().numpy(), y_pred.detach().numpy())
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    roc_curve_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
    roc_curve_display.plot()
    return roc_curve_display


# **Visualizing the plots:**

def viz(data: np.ndarray):
    pass


# Build a new neural network and try overfitting your training set


# **Generate data:**

def generate_data_overfit():
    return generate_data()


# **Define the Model:**

class OverfitModel(nn.Module):
    def __init__(self):
        super().__init__()


# **Training and validation:**

def train_overfit(model: nn.Module) -> nn.Module:
    return train_model(model=model)


# **Visualizing the plots:**

def viz_overfit(data):
    return viz(data)


"""
5. Briefly explain graph's results.
6. How does your metric value differs between the training data and the test data and why?
"""


def main():
    triple = generate_data()
    viz_data(*triple)


if __name__ == '__main__':
    main()
