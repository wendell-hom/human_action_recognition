#!/usr/bin/env python

#%%
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchsummary
import torchvision.transforms as transforms
from torch.utils import data

from usermodels import *
from utilities import *
from tqdm import tqdm
from PIL import Image
import pandas



#%%

project_name = "stanford40"

pixels = 224
rescale = int(pixels * 1.25)

batch_size = 16
test_batch_size = 64
num_workers = 4
num_epochs = 15

learning_rate = 0.0001
momentum = 0.9

device = torch.device("cuda:1")
train_path = "data/Stanford40/train/"
validation_path = "data/Stanford40/test/"

augmentation = transforms.Compose([
        transforms.Resize((rescale, rescale)),
        transforms.RandomCrop((pixels,pixels)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(train_path, transform=augmentation)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True )

test_data = torchvision.datasets.ImageFolder(validation_path, transform=augmentation)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, num_workers=num_workers, shuffle=True)


#%%

def train_test(start, end, history):
    train_acc, train_loss, val_acc, val_loss = history
    for n in range(start, end+1):

        train_acc, train_loss = train(model, train_loader, n, end, optimizer, loss_fn, device, train_acc, train_loss)
        val_acc, val_loss = test(model, test_loader, n, end, loss_fn, device, val_acc, val_loss)
    
    return train_acc, train_loss, val_acc, val_loss


#%%

model = torchvision.models.vgg16(pretrained=True)
model.to(device)

#%%


loss_fn = F.cross_entropy
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum, weight_decay =0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Wandb setup
# comment out if we don't want to use wandb
setup_wandb(project_name, model, batch_size, test_batch_size, num_epochs, learning_rate, momentum, log_interval=1)

history = ([], [], [], [])
train_acc, train_loss, val_acc, val_loss = train_test(1, num_epochs, history)

print("Finished training")

plot(train_acc, train_loss, val_acc, val_loss)


# %%

# Save model information
def save_model(filepath, epoch, model, optimizer, history):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history' : history
    }
    torch.save(state, filepath)