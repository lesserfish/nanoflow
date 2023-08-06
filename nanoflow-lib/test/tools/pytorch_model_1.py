import torch
import torch.nn as nn

class SimpleNeuralNetwork(nn.Module):
    def __init__(self, weights, bias):
        super(SimpleNeuralNetwork, self).__init__()
        self.linear = nn.Linear(3, 1)  # Three inputs, one output
        self.linear.weight.data = torch.tensor(weights).float()
        self.linear.bias.data = torch.tensor(bias).float()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Custom weights and bias
custom_weights = [[0.1, 0.2, 0.3]]  # A 1x3 matrix for a single output neuron
custom_bias = [0.4]  # A single value for the bias term

# Testing the model with custom weights and bias
input_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # Two samples with three inputs each
model = SimpleNeuralNetwork(custom_weights, custom_bias)
output = model(input_data)

print("Input:")
print(input_data)
print("Output:")
print(output)

