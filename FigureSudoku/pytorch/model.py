from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_layers=[64, 64]):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_layers (list): List containing the hidden layer sizes
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.model = nn.Sequential(
            nn.Linear(state_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, action_size)
        )
        # # Create an OrderedDict to store the network layers
        # layers = OrderedDict()
        #
        # # Include state_size and action_size as layers
        # hidden_layers = [state_size] + hidden_layers
        #
        # # Iterate over the parameters to create layers
        # for idx, (hl_in, hl_out) in enumerate(zip(hidden_layers[:-1], hidden_layers[1:])):
        #     # Add a linear layer
        #     layers['fc'+str(idx)] = nn.Linear(hl_in, hl_out)
        #     # Add an activation function
        #     layers['activation'+str(idx)] = nn.ReLU()
        #
        # # Create the output layer
        # layers['output'] = nn.Linear(hidden_layers[-1], action_size)
        #
        # # Create the network
        # self.network = nn.Sequential(layers)

        #self.linear1 = torch.nn.Linear(state_size, 512)
        #self.activation1 = torch.nn.ReLU()
        #self.linear2 = torch.nn.Linear(2048, 512)
        #self.activation2 = torch.nn.ReLU()
        #self.linear3 = torch.nn.Linear(512, action_size)
        #self.softmax = torch.nn.Softmax()

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        # Perform a feed-forward pass through the network
        #return self.network(state)

        #x = self.linear1(state)
        #x = self.activation1(x)
        #x = self.linear2(x)
        #x = self.activation2(x)
        #x = self.linear3(x)
        #x = self.softmax(x)
        return self.model(state)


class QNetworkCNN(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetworkCNN, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(state_size[-1], 4, 3, padding=1)
        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.conv3 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv5 = nn.Conv2d(16, 16, 3, padding=1)

        self.pool = nn.MaxPool2d(2, ceil_mode=True)

        self.fc1 = nn.Linear(3 * 3 * 16, 10)
        self.fc2 = nn.Linear(10, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        #print(state.shape)

        # Perform a feed-forward pass through the network
        x = self.pool(F.relu(self.conv1(state)))

        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
