import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

# Data Preprocessing
X = np.load('../Data/X_train_tabular.npy')
y = np.load('../Data/y_train_tabular.npy')

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# LinearModel
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(X_train.shape[1], 256),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.Dropout(0.3),
            nn.Linear(64, 14)  # Output shape (batch_size, 14)
        )

    def forward(self, x):
        return self.layers(x)

model = LinearModel()


# device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# loss function
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)


epochs = 10

train_losses = []
val_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y.view(-1, 14))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y.view(-1, 14))
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    scheduler.step(avg_val_loss)

    print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')


# plotting Training and Validation Loss
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.legend()
plt.show()
plt.savefig('LossDNN.png')

# Evaluation
model.eval()
test_loss = 0.0
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        outputs = model(batch_X)
        # Print shapes for debugging
        print(f'Test Output shape: {outputs.shape}, Test Target shape: {batch_y.shape}')
        loss = criterion(outputs, batch_y.view(-1, 14))
        test_loss += loss.item()

avg_test_loss = test_loss / len(val_loader)
print(f'Test Loss: {avg_test_loss:.4f}')



# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.preprocessing import StandardScaler
# from sklearn.model_selection import train_test_split
# import matplotlib.pyplot as plt
# import numpy as np
# from torchviz import make_dot  # Import torchviz
# import graphviz
# from torchviz import make_dot
# import os
#
# os.environ['PATH'] += r';C:\Program Files\Graphviz\bin'
# # Data Preprocessing
# X = np.load('../Data/X_train_tabular.npy')
# y = np.load('../Data/y_train_tabular.npy')
#
# scaler = StandardScaler()
# X = scaler.fit_transform(X)
#
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
#
# train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
# val_dataset = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32))
#
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
#
# # Specify the path to Graphviz's dot executable
# graphviz.backend.execute._find_executable = lambda: 'C:\\Program Files\\Graphviz\\bin\\dot.exe'
# # Model Definition
# class LinearModel(nn.Module):
#     def __init__(self):
#         super(LinearModel, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Linear(X_train.shape[1], 256),
#             nn.Dropout(0.3),
#             nn.Linear(256, 128),
#             nn.Dropout(0.3),
#             nn.Linear(128, 64),
#             nn.Dropout(0.3),
#             nn.Linear(64, 14)  # Output shape (batch_size, 14)
#         )
#
#     def forward(self, x):
#         return self.layers(x)
#
# model = LinearModel()
#
# # Your existing code
# dummy_input = torch.randn(1, X_train.shape[1])  # Dummy input tensor with the shape of one sample
# output = model(dummy_input)  # Forward pass with dummy input
# dot = make_dot(output, params=dict(model.named_parameters()))  # Create the computation graph
# dot.format = 'png'
# dot.render('model_architecture')  # Save the graph as a PNG file
