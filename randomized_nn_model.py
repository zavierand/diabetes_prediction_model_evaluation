# import necessary packages
import numpy as np
import torch
from torch import nn
from torch import optim

class RandomizedFNN(nn.Module):
    def __init__(self, X, y, numHiddenLayers = 3):
        super(RandomizedFNN, self).__init__()
        self.X = X
        self.y = y

        # parameters
        self.numFeatures = self.X.shape[1]

        # hyperparameters
        self.dense_nodes = self.numFeatures * 2
        self.learning_rate = 1e-2
        self.numHiddenLayers = numHiddenLayers

        activation_pool = [
            nn.ReLU(),
            nn.Tanh(),
            nn.Sigmoid(),
            nn.LeakyReLU(),
            nn.ELU()
        ]

        # input layer
        layers = []
        layers.append(nn.Linear(self.numFeatures, self.dense_nodes))
        layers.append(np.random.choice(activation_pool))

        # hidden layers
        for _ in range(self.numHiddenLayers - 1):
            layers.append(nn.Linear(self.dense_nodes, self.dense_nodes))
            layers.append(np.random.choice(activation_pool))

        # output layer
        layers.append(nn.Linear(self.dense_nodes, 1))  # Output for regression

        # define FNN
        self.fnn = nn.Sequential(*layers)

        # hyperparameters
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.fnn.parameters(), lr = self.learning_rate)

    def _train(self, num_epochs = 200):
        print(self.fnn) # see the activation functions used

        # training loop
        for epoch in range(0, num_epochs):
            self.fnn.train()

            # forward
            y_pred = self.fnn(self.X)
            self.y = self.y.view(-1, 1)

            loss = self.criterion(y_pred, self.y)

            # backwards
            self.optimizer.zero_grad()
            loss.backward()

            # update weights and biases
            self.optimizer.step()

            # output epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def test(self, X, y):
        self.fnn.eval()

        with torch.no_grad(): # disable gradient calculation
            y_pred = self.fnn(X)
            y = y.view(-1, 1)  # Ensure y has the shape [batch_size, 1]

            # compute loss
            loss = self.criterion(y_pred, y)
            
            # print output
            print(f'Test Loss: {loss.item():.6f}')