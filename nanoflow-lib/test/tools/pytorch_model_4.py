import torch
import torch.nn as nn

# Custom weights and bias
custom_weights = [[0.1, 0.2], [0.3, 0.4]]  # A 3x3 matrix for three output neurons
custom_bias = [0.4, 0.5]  # Three values for the bias term

# Training data and targets
inputs = torch.tensor([[1.0, 2.0]])
targets = torch.tensor([[10.0, 11.0]])

# Define the neural network with custom weights and bias
class SimpleNeuralNetwork(nn.Module):
    def __init__(self, weights, bias):
        super(SimpleNeuralNetwork, self).__init__()
        self.linear = nn.Linear(2, 2)  # Three inputs, three output neurons
        self.linear.weight.data = torch.tensor(weights).float()
        self.linear.bias.data = torch.tensor(bias).float()
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        return x

# Initialize the model and define the loss function
model = SimpleNeuralNetwork(custom_weights, custom_bias)
print("Feedforward: ")
print(model(torch.tensor([[1.0, 2.0]]) ))

criterion = nn.MSELoss()

# Training loop with one epoch
print("\n\nGrads:")
for inputs, targets in zip(inputs, targets):
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Update the weights manually using gradient descent
    with torch.no_grad():
        for param in model.parameters():
            print(param.grad)
            param -= param.grad


# Print the updated weights

print("\n\nUpdated Weights:")
print(model.linear.weight.data)
print("Updated Bias:")
print(model.linear.bias.data)

