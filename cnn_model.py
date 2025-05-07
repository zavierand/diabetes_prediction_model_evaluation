# import necessary packages
import numpy as np
import torch
from torch import nn
from torch import optim

# create a class to help clean up notebook
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self, X, y, in_ch, out_ch):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # instantiate our input and output
        self.X = X
        self.y = y

        # instantiate network parameters and hyperparameters
        self.classes = np.array([0, 1])
        self.in_ch = in_ch
        self.out_ch = out_ch

        # initialize dense layer neuron size
        self.num_features = X.shape[-1]  # set to last index in tensor (# samples, # channels, # features)
        self.learning_rate = 1e-2   # 0.001
        self.dense_neurons = self.num_features * 2  # hyperparameter

        # weights and biases
        # self.w = np.random(len(X.shape[1])) # random weights to start, which will be optimized with SGD
        # self.b = len(y.shape)   # we only need one layer of weights for each input

        # define the convolutional layers
        self.conv_layers = nn.Sequential(
            # first filtering layer
            nn.Conv1d(in_channels = 1, out_channels = 2, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 1, stride = 1),

            # second filtering layer
            nn.Conv1d(in_channels = 2, out_channels = 4, kernel_size = 3, stride = 1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size = 1, stride = 1)
        )

        # flatten features
        with torch.no_grad():
            # print(f'num features: {self.num_features}')
            dummy_input = torch.zeros(1, 1, self.num_features)
            out = self.conv_layers(dummy_input)
            # print(f"Shape after convolution: {out.shape}")
            self.flat_features = out.view(1, -1).shape[1]
            # print(f'flattened: {self.flat_features}')

        self.dense_neurons = self.flat_features * 2

        # define the forward feeding layers of feedforward network post convolution
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flat_features, self.dense_neurons),
            nn.ReLU(),

            # first hidden layer
            nn.Linear(self.dense_neurons, self.dense_neurons),
            nn.ReLU(),

            # second hidden layer
            nn.Linear(self.dense_neurons, self.dense_neurons),
            nn.ReLU(),

            # last hidden layer
            nn.Linear(self.dense_neurons, self.dense_neurons),
            nn.ReLU(),

            # output layer
            nn.Linear(self.dense_neurons, len(self.classes)),
            nn.LogSoftmax(dim=1)
        )

    # define how forward prop works
    def forward(self, X):
        if X.dim() == 2:
            # (batch_size, num_features) -> (batch_size, 1, num_features)
            X = X.unsqueeze(1)
        elif X.shape[1] != 1:
            # if shape is (batch, num_features, 1) --> permute
            X = X.permute(0, 2, 1)

        X = self.conv_layers(X)
        X = X.view(X.size(0), -1)  # flatten before fully connected layers
        X = self.fc_layers(X)
        return X

    # train method
    def _train(self, num_epochs):
        self.optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        self.criterion = nn.CrossEntropyLoss()
        self.losses = []

        # training loop
        for epoch in range(num_epochs):
            self.train()
            self.optimizer.zero_grad()

            # forward prop
            outputs = self.forward(self.X) 
            loss = self.criterion(outputs, self.y)

            # backprop & update weights
            loss.backward()
            self.optimizer.step()

            # output training progress
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            self.losses.append(loss.item())


    # test method
    def test(self, X_test, y_test):
        self.eval()  # set model to evaluation mode
        with torch.no_grad():
            outputs = self.forward(X_test)
            predicted = torch.argmax(outputs, dim=1)
            actual = torch.argmax(y_test, dim=1) if y_test.ndim > 1 else y_test
            y_pred_proba = torch.exp(outputs)[:, 1].cpu().numpy()  # get probability of positive class

            correct = (predicted == actual).sum().item()
            total = actual.size(0)
            accuracy = correct / total * 100

            print(f"Test Accuracy: {accuracy:.2f}%")
        return accuracy, predicted.cpu().numpy(), actual.cpu().numpy(), y_pred_proba