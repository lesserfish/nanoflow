import torch
import torch.nn as nn

# Custom weights and bias
custom_weights = [[0.1, 0.2, 0.3]]  # A 1x3 matrix for a single output neuron
custom_bias = [0.4]  # A single value for the bias term

# Training data and targets
inputs = torch.tensor([[1.0, 2.0, 3.0],[4.0, 5.0, 6.0]])
targets = torch.tensor([[10.0], [20.0]])

# Define the neural network with custom weights and bias
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

# Initialize the model and define the loss function
model = SimpleNeuralNetwork(custom_weights, custom_bias)
criterion = nn.MSELoss()

# Training loop with one epoch
for inputs, targets in zip(inputs, targets):
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    with torch.no_grad():
        for param in model.parameters():
            print(param.grad)
       
    # Update the weights manually using gradient descent
with torch.no_grad():
    for param in model.parameters():
        #print(param.grad)
        param -= param.grad

# Print the updated weights
print("Updated Weights:")
print(model.linear.weight.data)
print("Updated Bias:")
print(model.linear.bias.data)

