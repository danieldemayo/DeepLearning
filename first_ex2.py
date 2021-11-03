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
from torch import nn
from torch.autograd import Variable
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib import cm


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


# **Define the Model:**

class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()


# **Training and validation:**

def train_model(model: nn.Module) -> nn.Module:
    pass


def validate_model(model: nn.Module) -> list:
    pass


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
