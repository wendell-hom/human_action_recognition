#!/usr/bin/env python

#%%
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.transforms as transforms
from torch.utils import data

from usermodels import *
from utilities import *
from tqdm import tqdm
from PIL import Image
import pandas
from pathlib import Path
import glob

print(f"Torch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")

%load_ext autoreload
%autoreload 2
#%%

project_name = "stanford40"

pixels = 224
rescale = int(pixels * 1.25)

batch_size = 64
test_batch_size = 64
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

validation_data = torchvision.datasets.ImageFolder(validation_path, transform=test_transform)
validation_loader = torch.utils.data.DataLoader(validation_data, batch_size=test_batch_size, num_workers=num_workers, shuffle=True)

class_to_idx, idx_to_class = get_class_mappings(train_data)


#%%

def train_test(num_epochs, history, directory, early_stop, patience = math.inf):

    start = len(history)+1
    end = start + num_epochs

    for n in range(start, end):

        train_acc, train_loss = train(model, train_loader, n, end, optimizer, loss_fn, device)
        val_acc, val_loss = test(model, validation_loader, n, end, loss_fn, device)
        history.append_train_results(train_acc, train_loss)
        improvement = history.append_val_results(val_acc, val_loss)

        if improvement:
            save_model(directory, model, optimizer, history)
    
        history = early_stop(improvement, ckpt_directory, model, optimizer, history)

    return history


#%%

model_name = "MobileNet-V2"

device = torch.device("cuda:0")
restore_checkpoint = False
directory = f"models/Stanford40/{model_name}/"

if pixels == 224:
    ckpt_directory = os.path.join(directory, "checkpoints_224")
else:
    ckpt_directory = os.path.join(directory, "checkpoints_500")

Path(ckpt_directory).mkdir(parents=True, exist_ok=True)


if model_name == "VGG-16":
    model = torchvision.models.vgg16(pretrained=True)
    model.classifier[-1] = nn.Linear(4096, 40, True)
elif model_name == "VGG-19":
    model = torchvision.models.vgg19(pretrained=True)
    model.classifier[-1] = nn.Linear(4096, 40, True)
elif model_name == "ResNet-50":
    model = torchvision.models.resnet50(pretrained=True)
    model.fc = nn.Linear(2048, 40, True)
elif model_name == "MobileNet-V2":
    model = torchvision.models.mobilenet_v2(pretrained=True)
    model.classifier[-1] = nn.Linear(1280, 40, True)
elif model_name == "ResNeXt-50":
    model = torchvision.models.resnext50_32x4d(pretrained=True)
    model.fc = nn.Linear(2048, 40, True)
elif model_name == "ResNeXt-101":
    model = torchvision.models.resnext101_32x8d(pretrained=True)
    model.fc = nn.Linear(2048, 40, True)
elif model_name == "DenseNet-121":
    model = torchvision.models.densenet121(pretrained=True)
    model.classifier = nn.Linear(1024, 40, True)
else:
    model = None
 

model.to(device)

loss_fn = F.cross_entropy

history = History()

# try out early stop with rollback to rollback to epoch with the best weights and continue training
# set patience to infinity if we want to disable this and use regular training
early_stop = EarlyStopTraining(patience=3, retry_patience=6)

#optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=momentum, weight_decay =0.001)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

if restore_checkpoint:
    epoch, history = load_model(filepath, model, optimizer)


#%%

# Wandb setup
# comment out if we don't want to use wandb
#setup_wandb(project_name, model, batch_size, test_batch_size, num_epochs, learning_rate, momentum, log_interval=1)

train_test(1000, history, ckpt_directory, early_stop)
history.report_results()

#%% 
##### Additional Training ######

set_optimizer_lr(optimizer, learning_rate/10)
train_test(num_epochs, history, ckpt_directory)
history.report_results()

#%% 
##### Additional Training ######

set_optimizer_lr(optimizer, learning_rate/100)
train_test(num_epochs, history, ckpt_directory)
history.report_results()


# %%

##### Saving the model ######
save_model(f"{directory}/{pixels}_epoch_{len(history)}.pt", model, optimizer, history)



#%%
##### Evaluate on our own images #####

test_path = "data/Stanford40/images"
test_data = torchvision.datasets.ImageFolder(test_path, transform=test_transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=test_batch_size, num_workers=num_workers, shuffle=True)

#%%
# Confusion matrix
acc, loss, ground_truth, all_predictions = eval(model, validation_loader, loss_fn, device)

# Create and display the confusion matrix
matrix = metrics.confusion_matrix(ground_truth, all_predictions)

df_cm = pd.DataFrame(matrix, index = [idx_to_class[i] for i in range(40)], columns = [idx_to_class[i] for i in range(40)])
plt.figure(figsize = (40, 40))
sn.heatmap(df_cm, annot=True)


#%%

history = load_last_checkpoint(ckpt_directory, model, optimizer, history)
history = load_best_checkpoint(ckpt_directory, model, optimizer, history)
history.report_results()



# %%