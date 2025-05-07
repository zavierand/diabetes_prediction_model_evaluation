# import necessary packages
import numpy as np
import torch
from torch import nn
from torch import optim

class BinaryFNN(nn.Module):
    def __init__(self, X, y, activation_func, numHiddenLayers = 3):
        super(BinaryFNN, self).__init__()
        self.X = X
        self.y = y

        # parameters
        self.numFeatures = self.X.shape[1]

        # hyperparameters
        self.dense_nodes = self.numFeatures * 2
        self.learning_rate = 1e-2
        self.numHiddenLayers = numHiddenLayers
        self.activation_func = activation_func
        self.classes = [0, 1]

        # input layer
        layers = []
        layers.append(nn.Linear(self.numFeatures, self.dense_nodes))

        activation_pool = {
            'relu': nn.ReLU(),
            'tanh':nn.Tanh(),
            'sig': nn.Sigmoid(),
            'leakyRelu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }

        # hidden layers specified by hidden layers argument
        for _ in range(self.numHiddenLayers - 1):
            layers.append(nn.Linear(self.dense_nodes, self.dense_nodes))
            layers.append(activation_pool[self.activation_func])

        # n-1 hidden layer
        #layers.append(nn.Linear(self.dense_nodes, self.dense_nodes))
        #layers.append(nn.LogSoftmax(dim=1))

        # output layer
        layers.append(nn.Linear(self.dense_nodes, len(self.classes)))  # output for classification (singular val)

        # define FNN
        self.bfnn = nn.Sequential(*layers)

        # hyperparameters
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.bfnn.parameters(), lr = self.learning_rate)

    def _train(self, X, num_epochs = 200):
        # training loop
        for epoch in range(0, num_epochs):
            self.bfnn.train()

            # forward
            y_pred = self.bfnn(X)
            loss = self.criterion(y_pred, self.y)

            # backwards
            self.optimizer.zero_grad()
            loss.backward()

            # update weights and biases
            self.optimizer.step()

            # output epoch
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

    def test(self, X, y):
        self.bfnn.eval()

        with torch.no_grad(): # disable gradient calculation
            y_pred = self.bfnn(X)
            # compute loss
            loss = self.criterion(y_pred, self.y)
            
            print(f'Test Loss: {loss.item():.6f}')

            probs = torch.softmax(y_pred, dim=1)[:, 1].cpu().numpy()


            # also get predicted labels (optional)
            preds = torch.argmax(y_pred, dim=1).cpu().numpy()
            y_true = y.cpu().numpy()

            #return y_true, preds, probs