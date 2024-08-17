#%%
!pip install transformers torch numpy pandas scikit-learn matplotlib
#%%
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset,random_split
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
#%%
import torch.nn as nn
from transformers import BertModel, BertConfig

class AnglePredictor(nn.Module):
    def __init__(self):
        super(AnglePredictor, self).__init__()
        config = BertConfig(
            hidden_size=128,  
            num_attention_heads=2,  
            num_hidden_layers=2, 
            intermediate_size=128,  
            max_position_embeddings=12246  
        )

        self.embeddings = nn.Linear(12, 128)  
        self.transformer = BertModel(config)
        self.fc = nn.Linear(128, 14)  

    def forward(self, x):
        x = self.embeddings(x)  
        attention_mask = torch.ones(x.size()[:-1], device=x.device)  
        x = self.transformer(inputs_embeds=x, attention_mask=attention_mask)[0]
        x = self.fc(x)
        return x
#%%
# Load data
X_train = np.load('/kaggle/input/mydata/X_train_padding.npy')
y_train = np.load('/kaggle/input/mydata/y_train_padding.npy')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)
#%%
# Initialize the model
model = AnglePredictor()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
#%%
num_epochs = 10
train_losses = []
test_losses = []
#%%
from torch.utils.data import DataLoader, TensorDataset

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # Batch size of 1 due to sequence length

# Training loop
epochs = 20
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f'Epoch {epoch}/{epochs}, Loss: {epoch_loss/len(train_loader)}')
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}')
#%%
# Evaluating the model
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')

# Visualize some predictions
import matplotlib.pyplot as plt

plt.plot(y_test_tensor.numpy()[0][:, 0], label='True Angles')
plt.plot(predictions.numpy()[0][:, 0], label='Predicted Angles')
plt.legend()
plt.show()
#%% md
# 