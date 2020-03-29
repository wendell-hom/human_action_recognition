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

print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

#%%

project_name = "stanford40"

pixels = 500
rescale = int(pixels * 1.25)

batch_size = 8
test_batch_size = 32
num_workers = 4
num_epochs = 15

learning_rate = 0.0001
momentum = 0.9

device = torch.device("cuda:0")
train_path = "data/Stanford40/train/"
validation_path = "data/Stanford40/test/"

augmentation = transforms.Compose([
        transforms.Resize((rescale, rescale)),
        transforms.RandomCrop((pixels,pixels)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor()
])

test_transform = transforms.Compose([
        transforms.Resize((pixels, pixels)),
        transforms.ToTensor()
])

train_data = torchvision.datasets.ImageFolder(train_path, transform=augmentation)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True )

test_data = torchvision.datasets.ImageFolder(validation_path, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, num_workers=num_workers, shuffle=True)

class_to_idx, idx_to_class = get_class_mappings(train_data)


#%%

def train_test(start, end, history):
    train_acc, train_loss, val_acc, val_loss = history
    for n in range(start, end+1):

        train_acc, train_loss = train(model, train_loader, n, end, optimizer, loss_fn, device, train_acc, train_loss)
        val_acc, val_loss = test(model, test_loader, n, end, loss_fn, device, val_acc, val_loss)
    
    return train_acc, train_loss, val_acc, val_loss


#%%

model_name = "ResNet-50"

if model_name == "VGG-16":
    model = torchvision.models.vgg16(pretrained=True)
elif model_name == "VGG-19":
    model = torchvision.models.vgg19(pretrained=True)
elif model_name == "ResNet-50":
    model = torchvision.models.resnet50(pretrained=True)
elif model_name == "MobileNet-V2":
    model = torchvision.models.mobilenet_v2(pretrained=True)
else:
    model = None
 

device = torch.device("cuda:1")
model.to(device)

#%%


loss_fn = F.cross_entropy
#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum, weight_decay =0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Wandb setup
# comment out if we don't want to use wandb
#setup_wandb(project_name, model, batch_size, test_batch_size, num_epochs, learning_rate, momentum, log_interval=1)

history = ([], [], [], [])
train_acc, train_loss, val_acc, val_loss = train_test(1, num_epochs, history)

print("Finished training")

plot(train_acc, train_loss, val_acc, val_loss)

#%% 
##### Additional Training ######

set_optimizer_lr(optimizer, learning_rate/10)
train_acc, train_loss, val_acc, val_loss = train_test(16, 30, history)

plot(train_acc, train_loss, val_acc, val_loss)

# %%

##### Saving the model ######
save_model("models/Stanford40/vgg19/epoch_30.pt", num_epochs, model, optimizer, history)



#%%
##### Evaluate on our own images #####

test_path = "data/Stanford40/images"
test_data = torchvision.datasets.ImageFolder(test_path, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, num_workers=num_workers, shuffle=True)

acc, loss = eval(model, test_loader, idx_to_class, loss_fn, device)

#%%

