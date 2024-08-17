<div align="center" style="display: flex; justify-content: center;">
  <img src="https://github.com/user-attachments/assets/aa89aa8a-e29d-4149-bfcd-099daea57660" alt="racing" width="200" height="180" style="margin-right: 10px;" />
  <img src="https://github.com/user-attachments/assets/40d0678d-866a-4074-ac29-73b1d1a50082" alt="logo" width="200" height="180" style="margin-left: 10px;" />
  <h1>Deep Learning Team Mission</h1>
  <span style="font-size: 24px;">ASU Racing Team</span>
</div>


# Task Overview
- This project involves building and training various neural network models using tabular and sequential datasets. The models include linear (DNNs), CNN, RNN, CNN-LSTM, and attention-based architectures. Each model is trained for no more than 10 epochs. Additionally, the project includes data compression and representation with low dimensionality.

## Task 1: Model Training
### 1.1 Linear Model
- Dataset: Tabular
- Architecture: Deep Neural Networks (DNNs) with Linear (Fully Connected) Layers.
[Here](./EMG/models/DNN)
#### Assumptions:

- The tabular data is normalized.
- Linear layers can capture the relationships in tabular data.
#### Bottlenecks:

- Linear layers may not capture complex patterns compared to CNN or RNN.
### 1.2 CNN Model (Tabular Data)
- Dataset: Tabular
- Architecture: Convolutional Neural Networks (CNN) with Linear Layers.
[Here](./EMG/models/CNN_tabluar)
#### Assumptions:

- Using convolution layers to extract features from the tabular data.
#### Bottlenecks:

- Limited benefit from convolutions on tabular data as opposed to image or sequential data.
### 1.3 CNN Model (Sequential Data)
- Dataset: Sequential
- Architecture: CNN with time-based convolution (suitable time window).
[Here](./EMG/models/CNN_padding)
#### Assumptions:

- Temporal patterns can be captured by convolving over time.
#### Bottlenecks:

- Determining the optimal time window for convolutions can be challenging.
### 1.4 RNN Model
- Dataset: Sequential
- Architecture: Recurrent Neural Networks (RNN) with possible Linear Layers.
[Here](./EMG/models/RNN)
for Complex model [Here](./EMG/models/Complex_RNN)
#### Assumptions:

- RNNs are effective for capturing temporal dependencies in sequential data.
#### Bottlenecks:

- RNNs may suffer from vanishing gradients, making it hard to learn long-term dependencies.
### 1.5 CNN-LSTM Model
- Dataset: Sequential
- Architecture: Combination of CNN and LSTM layers.
[Here](./EMG/models/CNN_LSTM)
#### Assumptions:

- CNNs can capture spatial patterns, and LSTMs can capture temporal patterns.
#### Bottlenecks:

- Combining both architectures may increase computational complexity.
### 1.6 Attention-Based Model (Bonus)
- Dataset: Sequential
- Architecture: Transformer model with attention mechanism.
  - Transformer [Here](./EMG/models/Transformer)
  - BERT [Here](./EMG/models/BERT_MODEL)
#### Assumptions:

- Attention mechanism can capture global dependencies effectively.
#### Bottlenecks:

- Transformers require significant computational resources.

## Task 2: Data Compression and Representation
- Objective: Compress the data or represent it with low dimensional representation and plot it.
[Here](./EMG/representational%20learning)
## Task 3: Implementing Paper Methodology (Bonus)
- Objective: Review a specific paper and implement the described methodology to achieve the same results.
[Here](./oral%20disease%20classification)
