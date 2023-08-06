import torch
import torch.nn as nn

# Custom weights and bias
custom_weights = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]  # A 3x3 matrix for three output neurons
custom_bias = [0.4, 0.5, 0.6]  # Three values for the bias term

# Define the neural network with custom weights and bias
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, weights, bias):
        super(SimpleNeuralNetwork, self).__init__()
        self.linear = nn.Linear(3, 3)  # Three inputs, three output neurons
        self.linear.weight.data = torch.tensor(weights).float()
        print(self.linear.weight)
        self.linear.bias.data = torch.tensor(bias).float()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Initialize the model and define the loss function
input_data = torch.tensor([[1.0, 2.0, 3.0]])  # Two samples with three inputs each
model = SimpleNeuralNetwork(custom_weights, custom_bias)
output = model(input_data)

print("Input:")
print(input_data)
print("Output:")
print(output)

