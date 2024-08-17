# Angle Predictor Model
## Overview

This project implements a neural network model to predict angles using PyTorch and the `BERT transformer` architecture. The model is designed to process sequences and predict angles based on input data.

## Model Architecture
The AnglePredictor class is a PyTorch nn.Module that combines a linear embedding layer, a `BERT transformer model`, and a final fully connected layer for angle prediction. Below are the details of each component:

1. Embeddings Layer:

- Type: `nn.Linear`
- Input Dimension: 12
- Output Dimension: 128
- Description: Transforms the input data to a higher-dimensional space suitable for the transformer.
2. Transformer Layer:

- Type: `BertModel` from the `transformers` library
- Configuration:
`hidden_size`: 128
`num_attention_heads`: 2
`num_hidden_layers`: 2
`intermediate_size`: 128
`max_position_embeddings`: 12246
- Description: Applies BERT transformer architecture for sequence processing.
3. Fully Connected Layer:

- Type: `nn.Linear`
- Input Dimension: 128
- Output Dimension: 14
- Description: Outputs the predicted angle values.
## Hyperparameters
- Learning Rate: 1e-4
- Batch Size: 1 (due to sequence length)
- Number of Epochs: 20
## Data Preparation
- Training Data: Loaded from `X_train_padding.npy` and `y_train_padding.npy`
- Testing Data: Split from the training data.

Data is converted to PyTorch tensors for training and evaluation.

## Training and Evaluation
The training loop consists of 20 epochs, with the loss computed using Mean Squared Error (MSE). The model is evaluated on a separate test set to compute the test loss.


### Training Code
```python
for epoch in range(epochs):
    epoch_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader)}')
```
### Evaluation Code
```python
model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor)
    test_loss = criterion(predictions, y_test_tensor)
    print(f'Test Loss: {test_loss.item()}')
```