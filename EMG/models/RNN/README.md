# RNN Model Implementation

<div align="center">
  <a href="model_architecture.png">
    <img src="model_architecture.png" alt="Logo" width="800" height="800">
  </a>

</div>
## Overview
This project implements a simple Recurrent Neural Network `(RNN)` using PyTorch for a regression task. The model consists of an RNN layer followed by two fully connected layers.


## Data Preparation
The data is loaded and preprocessed as follows:

1. Data Files:

   - `X_train_padding.npy`: Features
   - `y_train_padding.npy`: Target values
2. Preprocessing Steps:

   - Convert numpy arrays to PyTorch tensors.
   - Create a TensorDataset and DataLoader.
```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load and preprocess data
X = np.load('/kaggle/input/mydata/X_train_padding.npy')
Y = np.load('/kaggle/input/mydata/y_train_padding.npy')

# Convert numpy arrays to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

# Create a TensorDataset and DataLoader
dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```
## Model Architecture
The `RNN` class defines a simple recurrent neural network:

1. Layers:
   - `RNN Layer`: Input size of `12`, hidden size of `128`, and `2` layers
   - `Fully Connected Layers`:
     - Linear transformation from `128` to `64`
     - Linear transformation from `64` to `14`
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(input_size=12, hidden_size=128, num_layers=2, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 14)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 ```       
## Training
The model is trained with the following setup:

- Loss Function: Mean Squared Error (`nn.MSELoss`)
- Optimizer: Adam with a learning rate of `0.001`
- Batch Size: `8`
- Epochs: `10`
```python
import torch.optim as optim
import matplotlib.pyplot as plt

# Initialize the model, loss function, and optimizer
model = RNN()
if torch.cuda.is_available():
    model = model.to('cuda')

criterion = nn.MSELoss()  # Assuming a regression problem
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
all_losses = []

# Training loop
for epoch in range(num_epochs):
    epoch_loss = 0.0

    for batch_data, batch_target in dataloader:
        if torch.cuda.is_available():
            batch_data, batch_target = batch_data.to('cuda'), batch_target.to('cuda')
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_target)  # Ensure target shape matches output shape
        loss.backward()
        optimizer.step()

        # Accumulate the loss for the current batch
        epoch_loss += loss.item()

    # Calculate average loss for the epoch
    avg_epoch_loss = epoch_loss / len(dataloader)

    # Print the average loss for the epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Avg. Loss: {avg_epoch_loss:.4f}')

    # Save the average loss for plotting
    all_losses.append(avg_epoch_loss)

# Plot the training loss
plt.plot(all_losses, label='Training Loss')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.show()
plt.savefig('CNN_LSTM_training_loss.png')
```