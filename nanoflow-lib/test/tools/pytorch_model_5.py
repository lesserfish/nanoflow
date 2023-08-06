import torch
import torch.nn as nn

# Training data and targets
inputs = torch.tensor([[1.0, 2.0]])
targets = torch.tensor([[0.3]])

# Define the neural network
class ComplexNeuralNetwork(nn.Module):
    def __init__(self):
        super(ComplexNeuralNetwork, self).__init__()
        self.hidden_layer = nn.Linear(2, 64)  # Two input nodes, 64 hidden nodes
        self.output_layer = nn.Linear(64, 1)  # 64 hidden nodes, 1 output node
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        # Initialize weights and biases to 0.1
        nn.init.constant_(self.hidden_layer.weight, 0.1)
        nn.init.constant_(self.hidden_layer.bias, 0.1)
        nn.init.constant_(self.output_layer.weight, 0.1)
        nn.init.constant_(self.output_layer.bias, 0.1)

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

# Initialize the model and define the loss function
model = ComplexNeuralNetwork()
criterion = nn.MSELoss()

outputs = model(torch.tensor([[1.0, 2.0]]))
print("Output before: {}".format(outputs))

# Training loop with one epoch
total_loss = 0.0
for inputs, targets in zip(inputs, targets):
    model.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    loss.backward()
    
    # Update the weights manually using gradient descent
    with torch.no_grad():
        for param in model.parameters():
            param -= 1 * param.grad


outputs = model(torch.tensor([[1.0, 2.0]]))
print("Output after: {}".format(outputs))
