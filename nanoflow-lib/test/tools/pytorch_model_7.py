import torch
import torch.nn as nn
import torch.optim as optim

# Training data and targets
inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
targets = torch.tensor([[0.5], [0.8]])

# Define the neural network
class ComplexNeuralNetwork(nn.Module):
    def __init__(self):
        super(ComplexNeuralNetwork, self).__init__()
        self.hidden_layer = nn.Linear(2, 64)  # Two input nodes, 64 hidden nodes
        self.output_layer = nn.Linear(64, 1)  # 64 hidden nodes, 1 output node
        
        # Initialize weights and biases to 0.1
        nn.init.constant_(self.hidden_layer.weight, 0.1)
        nn.init.constant_(self.hidden_layer.bias, 0.1)
        nn.init.constant_(self.output_layer.weight, 0.1)
        nn.init.constant_(self.output_layer.bias, 0.1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden_layer(x)
        x = self.relu(x)
        x = self.output_layer(x)
        x = self.sigmoid(x)
        return x

# Initialize the model and define the loss function and optimizer
model = ComplexNeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1)

outputs = model(torch.tensor([[1.0, 2.0]]))
print("Output before: {}".format(outputs))

# Training loop with two epochs
epochs = 1

for epoch in range(epochs):
    total_loss = 0.0
    optimizer.zero_grad()  # Zero the gradients for each epoch
    outputs = model(inputs)  # Forward pass for all samples

    loss = criterion(outputs, targets)
    loss.backward()  # Compute gradients for all samples

    optimizer.step()  # Update the weights using the accumulated gradients

    total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss:.4f}")

outputs = model(torch.tensor([[1.0, 2.0]]))
print("Output after: {}".format(outputs))
