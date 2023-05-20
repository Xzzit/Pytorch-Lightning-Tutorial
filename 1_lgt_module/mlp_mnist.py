import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import pytorch_lightning as pl

# Create a dataset for training, validation and testing
mnist = torchvision.datasets.MNIST(root='./data', train=True, 
                                   transform=transforms.ToTensor(), download=True)
mnist_test = torchvision.datasets.MNIST(root='./data', train=False, 
                                   transform=transforms.ToTensor(), download=True)

# Split the dataset into train, validation and test
train_size = int(0.8 * len(mnist))
val_size = len(mnist) - train_size
train_data, val_data = random_split(mnist, [train_size, val_size])

# Create a dataloader for train, validation and test
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size=32, shuffle=True)
    
# Define the pytorch lightning module
class Model(pl.LightningModule):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = nn.Linear(28*28, 256)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(256)
        self.linear2 = nn.Linear(256, 256)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm1d(256)
        self.linear3 = nn.Linear(256, 128)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm1d(128)
        self.linear4 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = self.linear1(x)
        x = self.relu1(x)
        x = self.bn1(x)
        x = self.linear2(x)
        x = self.relu2(x)
        x = self.bn2(x)
        x = self.linear3(x)
        x = self.relu3(x)
        x = self.bn3(x)
        x = self.linear4(x)
        return x
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        loss = F.cross_entropy(scores, y)
        self.log('test_loss', loss)
        return loss
    
    def predict_step(self, batch, batch_idx):
        x, y = batch
        scores = self.forward(x)
        predictions = torch.argmax(scores, dim=1)
        return predictions

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=1e-3)

# Hyperparameters
learning_rate = 0.001
num_epochs = 10
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize the model
model = Model().to(device)
# torch.set_float32_matmul_precision('high')
# model = torch.compile(model)

# Create CUDA events for measuring time
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

# Measure the time for training
start_event.record()

# Train the model
trainer = pl.Trainer(accelerator='gpu', devices=[0], min_epochs=1, max_epochs=3, precision='16-mixed')
trainer.fit(model, train_loader, val_loader)

# Measure the time for training
end_event.record()
torch.cuda.synchronize()  # Wait for the events to be recorded!

# Calculate the training time
elapsed_time_ms = start_event.elapsed_time(end_event) / 1000
print(f"Training time: {elapsed_time_ms:.2f}s")

# Check accuracy on training & test to see how good our model
def check_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)
            
            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
            
        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
    
    model.train()

print("Accuracy on training set: ")
check_accuracy(train_loader, model)

print("Accuracy on testing set: ")
check_accuracy(test_loader, model)