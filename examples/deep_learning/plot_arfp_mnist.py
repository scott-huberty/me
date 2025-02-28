"""
Compute Accuracy, Precision, and Recall for a multi-class classification problem
================================================================================

The MNIST dataset has 10 categories. How do we adopt our classification performance
metrics to handle multi-class classification problems?
"""

# %%
# import libraries
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# %%
# Load the MNIST dataset
# -----------------------
#

# %%
digits = load_digits()
# Extract the data and labels
labels = digits.target
data = digits.data
# Constrain the values of the data to be between 0 and 1
data_normalized = data / np.max(data)

# %%
# Convert the data to PyTorch tensors
data_tensor = torch.tensor(data_normalized, dtype=torch.float32)
labels_tensor = torch.tensor(labels, dtype=torch.long)

# Split the data into training and testing sets
train_data, test_data, train_labels, test_labels = train_test_split(
    data_tensor, labels_tensor, test_size=.1, random_state=42
)

# Convert the data to PyTorch Datasets
train_data_ds = torch.utils.data.TensorDataset(train_data, train_labels)
test_data_ds = torch.utils.data.TensorDataset(test_data, test_labels)

# Convert the data to PyTorch DataLoaders objects
batch_size = 32
train_loader = DataLoader(train_data_ds, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(test_data_ds, batch_size=batch_size, shuffle=False, drop_last=False)

# %%
# Define the neural network
# -------------------------
#

# %%
class DigitsNet(nn.Module):
    def __init__(self):
        super(DigitsNet, self).__init__()

        # Define the layers
        self.input = nn.Linear(64, 32)
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 16)
        self.output = nn.Linear(16, 10)

    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)

# %%
digits_net = DigitsNet()
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(digits_net.parameters(), lr=.01)

# %%
# Train the neural network
# ------------------------
#

# %%
n_epochs = 10
losses = torch.zeros(n_epochs)
train_accuracies = []
test_accuracies = []

for ei in range(n_epochs):
    digits_net.train() # set the model to training mode

    batch_accuracies = []
    batch_losses = []
    for data, labels in train_loader:
        output = digits_net(data)
        loss = loss_fn(output, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # This batches loss
        batch_losses.append(loss.item())
        # Compute this batches accuracy
        matches = torch.argmax(output, dim=1) == labels
        matches_float = matches.float() # go from True/False to 1/0
        accuracy_percentage = 100 * matches_float.mean()
        batch_accuracies.append(accuracy_percentage)

    # Compute the average loss and accuracy across batches for this epoch
    losses[ei] = np.mean(batch_losses)
    train_accuracies.append(np.mean(batch_accuracies))

    # Compute the accuracy on the test set
    test_data, test_labels = next(iter(test_loader))
    with torch.no_grad():
        test_output = digits_net(test_data)
    matches = torch.argmax(test_output, dim=1) == test_labels
    matches_float = matches.float()
    test_accuracy_percentage = 100 * matches_float.mean()
    test_accuracies.append(test_accuracy_percentage)

# %%
# Plot the results
# ----------------
#

# %%
fig, ax = plt.subplots(1, 2, figsize=(10, 3), constrained_layout=True)

ax[0].plot(losses, label='Training Loss')
ax[0].set_xlabel('Epoch')
ax[0].set_ylabel('Loss')
ax[0].legend()

ax[1].plot(train_accuracies, label='Training Accuracy')
ax[1].plot(test_accuracies, label='Test Accuracy')
ax[1].set_xlabel('Epoch')
ax[1].set_ylabel('Accuracy (%)')
ax[1].legend()

plt.show()

# %%
# Compute Accuracy, Precision, and Recall
# ---------------------------------------
#
# Unlike the previous example (on the wine dataset), the MNIST dataset has 10 categories,
# so there are 10 accuracy/precision/recall values to compute.
# We have 3 options:
# - Compute the metrics for each class separately
# - Compute the metrics for each class and average them (unweighted average)
# - Compute the metrics for each class and average them (weighted average)
#

# %%
y_hat = digits_net(train_loader.dataset.tensors[0])
train_predictions = torch.argmax(y_hat, dim=1)

y_hat = digits_net(test_loader.dataset.tensors[0])
test_predictions = torch.argmax(y_hat, dim=1)

precision = precision_score(
    test_loader.dataset.tensors[1].detach().numpy(),
    test_predictions.detach().numpy(),
    average=None
)
recall = recall_score(
    test_loader.dataset.tensors[1],
    test_predictions,
    average=None
)

fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)

ax.bar(
    np.arange(10) - .2, precision, width=.4, label='Precision'
)
ax.bar(
    np.arange(10) + .2, recall, width=.4, label='Recall'
)
ax.set_xticks(np.arange(10))
ax.set_xlabel('Category')
ax.set_title('Precision and Recall for each category')
ax.legend()
plt.show()

# %%
# Compute the average precision and recall
# ----------------------------------------
#
# For the training and testing sets
#

# %%
fig, ax = plt.subplots()

train_metrics = [0, 0, 0 ,0]
test_metrics = [0, 0, 0, 0]

train_metrics[0] = accuracy_score(train_loader.dataset.tensors[1], train_predictions)
train_metrics[1] = precision_score(train_loader.dataset.tensors[1], train_predictions, average='macro')
train_metrics[2] = recall_score(train_loader.dataset.tensors[1], train_predictions, average='macro')
train_metrics[3] = f1_score(train_loader.dataset.tensors[1], train_predictions, average='macro')

test_metrics[0] = accuracy_score(test_loader.dataset.tensors[1], test_predictions)
test_metrics[1] = precision_score(test_loader.dataset.tensors[1], test_predictions, average='macro')
test_metrics[2] = recall_score(test_loader.dataset.tensors[1], test_predictions, average='macro')
test_metrics[3] = f1_score(test_loader.dataset.tensors[1], test_predictions, average='macro')

ax.bar(
    np.arange(4) - .2, train_metrics, width=.4, label='Training Set'
)
ax.bar(
    np.arange(4) + .2, test_metrics, width=.4, label='Test Set'
)
ax.set_xticks(np.arange(4))
ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1'])
ax.set_title('Metrics for the training and testing sets')
ax.legend()
plt.show()



