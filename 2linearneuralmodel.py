# Import necessary libraries
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)  # 100 points between 0 and 10
y = 2.5 * X + 5 + np.random.randn(100)  # Linear relation with noise

# Convert data to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32).view(-1, 1)
y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1)

# Define a simple linear regression model
class LinearRegressionModel(nn.Module):
    def _init_(self):
        super(LinearRegressionModel, self)._init_()
        self.linear = nn.Linear(1, 1)  # Single input and single output

    def forward(self, x):
        return self.linear(x)

# Initialize model, define loss function and optimizer
model = LinearRegressionModel()
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Training the model
epochs = 500
losses = []

for epoch in range(epochs):
    # Forward pass
    predictions = model(X_tensor)
    loss = criterion(predictions, y_tensor)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Record the loss
    losses.append(loss.item())
    
    # Print loss every 50 epochs
    if (epoch+1) % 50 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}")

# Plot the loss over epochs
plt.figure(figsize=(10, 6))
plt.plot(losses)
plt.title("Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid()
plt.show()

# Plot the predictions vs actual data
predicted = model(X_tensor).detach().numpy()

plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Actual Data", color="blue")
plt.plot(X, predicted, label="Predicted Line", color="red")
plt.title("Linear Regression with Neural Model")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()