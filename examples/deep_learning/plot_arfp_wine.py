"""

Computer Accuracy, Precision, Recaall, and F1 on the Wine dataset
=================================================================

Let's practice computing the accuracy, precision, recall, and F1 score on the Wine dataset.
We'll also use Scikit Learn to compute these metrics for us.
"""

#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# for number-crunching
import numpy as np

# for dataset management
import polars as pl

# for data visualization
import matplotlib.pyplot as plt
import matplotlib_inline.backend_inline
import seaborn as sns

# %%
# Load the Wine dataset
# ---------------------
# We'll use the Wine dataset from the UCI Machine Learning Repository.
# The dataset has 178 samples with 13 features each.
# The features are chemical properties of the wines.
# The target variable is the wine class (1, 2, or 3).

# %%
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
df = pl.read_csv(url, separator=";", infer_schema_length=int(1e5))
df = df.rename(lambda col_name : col_name.replace(" ", "_"))
# Drop a few outliers
df = df.filter(df["total_sulfur_dioxide"] < 200)

z_scores = [
    (pl.col(col) - pl.col(col).mean()) / pl.col(col).std()
    for col in df.columns
    if col != "quality"
    ]
df = df.select([
    pl.col("quality"),
    *z_scores
])

# create a new column for binarized (boolean) quality
df = df.with_columns(
    pl.when(df["quality"] > 5).then(1).otherwise(0).alias("good_quality")
)
df

# %%
# Convert to torch tensors
# ------------------------
#

# %%
train_tensor = df.select(
    [col for col in df.columns if col not in ["quality", "good_quality"]]
).to_torch().float()
labels_tensor = df.select("good_quality").to_torch().float()
print(train_tensor.shape, labels_tensor.shape)

# %%
# Split the data
# --------------
# We'll plit the data into training and testing sets using Scikit Learn

# %%
train_data, test_data, train_labels, test_labels = train_test_split(train_tensor, labels_tensor, test_size=.1)
# then convert them into PyTorch Datasets (note: already converted to tensors)
train_dataset = torch.utils.data.TensorDataset(train_data, train_labels)
test_dataset = torch.utils.data.TensorDataset(test_data, test_labels)

# Finally, create the DataLoader objects
n_samples = test_dataset.tensors[0].shape[0] 
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=n_samples, shuffle=False)

# %%
# Define the model
# ----------------
#

# %%
class WineNet(nn.Module):
  def __init__(self):
    super().__init__()

    ### input layer
    self.input = nn.Linear(11,16)
    
    ### hidden layers
    self.fc1 = nn.Linear(16,32)
    self.fc2 = nn.Linear(32,32)

    ### output layer
    self.output = nn.Linear(32,1)
  
  # forward pass
  def forward(self,x):
    x = F.relu( self.input(x) )
    x = F.relu( self.fc1(x) )
    x = F.relu( self.fc2(x) )
    return self.output(x)

# %%
# Define a function to train the model
# ------------------------------------
#

# %%
def train_the_model(
    wine_net,
    train_loader,
    test_loader,
    num_epochs=1_000,
    ):
  """Train the model on the Wine dataset.

  Parameters
  ----------
  wine_net : WineNet
    The neural network model defined above
  num_epochs : int
    The number of epochs to train the model
  """
  # loss function and optimizer
  lossfun = nn.BCEWithLogitsLoss()
  optimizer = torch.optim.SGD(wine_net.parameters(),lr=.01)

  # initialize losses
  losses   = torch.zeros(num_epochs)
  train_accuracies = []
  test_accuracies  = []

  # loop over epochs
  for epochi in range(num_epochs):

    # loop over training data batches
    batch_accuracies  = []
    batch_losses = []
    for x, y in train_loader:
      # forward pass and loss
      y_hat = wine_net(x)
      loss = lossfun(y_hat , y)

      # backprop
      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      # loss from this batch
      batch_losses.append(loss.item())

      # compute training accuracy for this batch
      batch_accuracies.append( 100 * torch.mean(((y_hat > 0) == y).float()).item() )
    # end of batch loop...

    # now that we've trained through the batches, get their average training accuracy
    train_accuracies.append( np.mean(batch_accuracies) )

    # and get average losses across the batches
    losses[epochi] = np.mean(batch_losses)

    # test accuracy
    x, y = next(iter(test_loader)) # extract X, y from test dataloader
    with torch.no_grad(): # deactivates autograd
      y_hat = wine_net(x)
    test_accuracies.append( 100 * torch.mean(((y_hat > 0) == y).float()).item() )
  
  # function output
  return train_accuracies, test_accuracies, losses

# %%
# Train the model
# ---------------
#

# %%
wine_net = WineNet()
train_accuracies, test_accuracies, losses = train_the_model(
  wine_net,
  train_loader,
  test_loader,
  )

# %%
# Compute the accuracy, precision, recall, and F1 score on the train and test sets
# ----------------------------------------------------------------------------------
#

# %%
train_predictions = wine_net(train_loader.dataset.tensors[0])
test_predictions = wine_net(test_loader.dataset.tensors[0])
test_predictions

# %%
# Use Scikit Learn to compute the metrics
# ---------------------------------------
#

# %%
from sklearn.metrics import (
  accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
  )

# initialize a dictionary to store the metrics
train_metrics = [0, 0, 0, 0]
test_metrics = [0, 0, 0, 0]

# compute the metrics on the train set
true_labels = train_loader.dataset.tensors[1]
train_predictions = train_predictions > 0
train_metrics[0] = accuracy_score(true_labels, train_predictions)
train_metrics[1] = precision_score(true_labels, train_predictions)
train_metrics[2] = recall_score(true_labels, train_predictions)
train_metrics[3] = f1_score(true_labels, train_predictions)

# compute the metrics on the test set
true_labels = test_loader.dataset.tensors[1]
test_predictions = test_predictions > 0
test_metrics[0] = accuracy_score(true_labels, test_predictions)
test_metrics[1] = precision_score(true_labels, test_predictions)
test_metrics[2] = recall_score(true_labels, test_predictions)
test_metrics[3] = f1_score(true_labels, test_predictions)
print(train_metrics, test_metrics)

# %%
# Plot the metrics
# ----------------
#

# %%
sns.set_style("darkgrid")
fig, ax = plt.subplots()
ax.bar(np.arange(4) -.1, train_metrics, .5)
ax.bar(np.arange(4) +.1, test_metrics, .5)
ax.set_xticks([0,1,2,3],['Accuracy','Precision','Recall','F1-score'])
ax.set_ylim([.6,1])
ax.legend(['Train','Test'])
ax.set_title('Performance metrics')
plt.show()

# %%
# Show the confusion matrices
# ---------------------------
#

# %%
# Confusion matrices
true_labels_train = train_loader.dataset.tensors[1]
true_labels_test = test_loader.dataset.tensors[1]
train_conf = confusion_matrix(true_labels_train, train_predictions>0)
test_conf  = confusion_matrix(true_labels_test, test_predictions>0)

sns.set_style('white')
fig, axes = plt.subplots(1,2,figsize=(10,4))

# confmat during TRAIN
axes[0].imshow(train_conf, 'Blues', vmax=len(train_predictions)/2)
axes[0].set_xticks([0,1])
axes[0].set_yticks([0,1])
axes[0].set_xticklabels(['bad','good'])
axes[0].set_yticklabels(['bad','good'])
axes[0].set_xlabel('Predicted quality')
axes[0].set_ylabel('True quality')
axes[0].set_title('TRAIN confusion matrix')

# add text labels
text_kwargs = dict(ha='center', va='center')
axes[0].text(0,0,f'True negatives:\n{train_conf[0, 0]}' , **text_kwargs)
axes[0].text(0,1,f'False negatives:\n{train_conf[1, 0]}', **text_kwargs)
axes[0].text(1,1,f'True positives:\n{train_conf[1, 1]}' , **text_kwargs)
axes[0].text(1,0,f'False positives:\n{train_conf[0, 1]}', **text_kwargs)

# confmat during TEST
axes[1].imshow(test_conf,'Blues',vmax=len(test_predictions)/2)
axes[1].set_xticks([0,1])
axes[1].set_yticks([0,1])
axes[1].set_xticklabels(['bad','good'])
axes[1].set_yticklabels(['bad','good'])
axes[1].set_xlabel('Predicted quality')
axes[1].set_ylabel('True quality')
axes[1].set_title('TEST confusion matrix')

# add text labels
axes[1].text(0,0,f'True negatives:\n{test_conf[0,0]}', **text_kwargs)
axes[1].text(0,1,f'False negatives:\n{test_conf[1,0]}', **text_kwargs)
axes[1].text(1,1,f'True positives:\n{test_conf[1,1]}' , **text_kwargs)
axes[1].text(1,0,f'False positives:\n{test_conf[0,1]}', **text_kwargs)
plt.show()