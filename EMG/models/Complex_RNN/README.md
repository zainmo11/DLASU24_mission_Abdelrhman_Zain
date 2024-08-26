# Complex RNN Model

<div align="center">
  <a href="model_architecture.png">
    <img src="model_architecture.png" alt="Logo" width="800" height="800">
  </a>

</div>

## Overview
This project implements a Recurrent Neural Network `(RNN)` using PyTorch for a regression task. The model features a bidirectional RNN with dropout for regularization and two fully connected layers for output prediction.


## Data Preparation
The data is loaded from local files. Ensure that you have the necessary .npy files available.

### Data Files

- `X_train_padding.npy`: Input features
- `y_train_padding.npy`: Target values

```python
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

X = np.load('/kaggle/input/mydata/X_train_padding.npy')
Y = np.load('/kaggle/input/mydata/y_train_padding.npy')

X_tensor = torch.tensor(X, dtype=torch.float32)
Y_tensor = torch.tensor(Y, dtype=torch.float32)

dataset = TensorDataset(X_tensor, Y_tensor)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
```

## Model Architecture
The ComplexRNN class defines the RNN model with the following architecture:

1. Recurrent Layer:

   - RNN: `nn.RNN` with `input_size=12`, `hidden_size=128`, `num_layers=3`, `batch_first=True`, `bidirectional=True`, and `dropout=0.5`
2. Fully Connected Layers:

   - fc1: `nn.Linear` with `in_features=128 * 2` (due to bidirectional RNN) and `out_features=64`
   - fc2: `nn.Linear` with `in_features=64` and `out_features=14`
3. Dropout Layer:

   - Dropout Rate: 0.5
``` python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ComplexRNN(nn.Module):
    def __init__(self):
        super(ComplexRNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=12, 
            hidden_size=128, 
            num_layers=3, 
            batch_first=True, 
            bidirectional=True,
            dropout=0.5
        )
        self.fc1 = nn.Linear(128 * 2, 64)  # Multiply hidden_size by 2 for bidirectional
        self.fc2 = nn.Linear(64, 14)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(x)  # Apply dropout
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
```        
## Training
The model is trained with the following settings:

- Loss Function: Mean Squared Error (nn.MSELoss)
- Optimizer: Adam with a learning rate of 0.001
- Batch Size: 8
- Epochs: 10

```python
Copy code
import torch.optim as optim

model = ComplexRNN()
if torch.cuda.is_available():
    model = model.to('cuda')
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
all_losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_data, batch_target in dataloader:
        if torch.cuda.is_available():
            batch_data, batch_target = batch_data.to('cuda'), batch_target.to('cuda')
        optimizer.zero_grad()
        outputs = model(batch_data)
        loss = criterion(outputs, batch_target)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Avg. Loss: {avg_epoch_loss:.4f}')
    all_losses.append(avg_epoch_loss)

# Plotting Training Loss
import matplotlib.pyplot as plt

plt.plot(all_losses, label='Training Loss')
plt.savefig('ComplexRNN.png')
plt.title('Training Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.legend()
plt.show()
```