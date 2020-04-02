#!/usr/bin/env

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from apex import amp
import math

import pandas as pd
import seaborn as sn
import os
from sklearn import metrics
import glob
import pdb

class History():
    def __init__(self):
        # Where to save the models
        self.train_acc, self.train_loss = [], []
        self.val_acc, self.val_loss = [], []


        #                epoch,  acc/loss
        self.best_acc = (None, -math.inf)
        self.best_loss = (None, math.inf)

    def __len__(self):
        return len(self.train_acc)

    def append_train_results(self, train_acc, train_loss):
        self.train_acc.append(train_acc)
        self.train_loss.append(train_loss)

    def append_val_results(self, val_acc, val_loss):
        self.val_acc.append(val_acc)
        self.val_loss.append(val_loss)

        epoch = len(self.val_acc)
        best_model = False

        if val_acc > self.best_acc[1]:
            self.best_acc = (epoch, val_acc)
            #best_model = True
        if val_loss < self.best_loss[1]:
            self.best_loss = (epoch, val_loss)
            best_model = True

        return best_model


    def get_best_loss(self):
        return self.best_loss

    def report_results(self):
        if len(self.train_acc) == 0:
            print("No results to report")
            return

        plot(self)
        epoch_acc = self.best_acc[0]
        epoch_loss = self.best_loss[0]

        print(f'Best Accuracy at Epoch {epoch_acc} val_acc: {self.best_acc[1]:.4f}, val_loss: {self.val_loss[epoch_acc-1]:.4f}')
        print(f'Best Loss     at Epoch {epoch_loss} val_acc: {self.val_acc[epoch_loss-1]:.4f}, val_loss: {self.best_loss[1]:.4f}')
        print(f'Total epochs: {len(self.train_acc)}')


class EarlyStopTraining():
    def __init__(self, patience, retry_patience):
        """ patience is the number of epochs to keep training without seeing loss or accuracy go down 
            retry patience is the number of times we reload to the previous checkpoint before decreasing learning rate 
        """

        self.patience = patience
        self.retry_patience = retry_patience

        self.no_improvement = 0
        self.num_retries = 0

    def __call__(self, best_model, ckpt_directory, model, optimizer, history):
        if best_model:
            self.no_improvement = 0
            self.num_retries = 0
        else:
            self.no_improvement += 1
        
        if self.no_improvement > self.patience:
            print("num retries: ", self.num_retries)
            history = load_best_checkpoint(ckpt_directory, model, optimizer, history)
            self.num_retries += 1
            print("num retries: ", self.num_retries)
            self.no_improvement = 0

            epoch, loss = history.get_best_loss()
            print(f"Best loss so far {loss} at epoch {epoch}")

            if self.num_retries > self.retry_patience:
                # decrease learning rate by a factor of 5
                lr = optimizer.param_groups[0]['lr'] / 5
                set_optimizer_lr(optimizer, lr)
                self.num_retries = 0
                print("Reducing learning rate to ", lr)

        return history 



def train(model, train_loader, epoch, epochs, optimizer, loss_fn, device, half_precision=False):
    model.train()

    running_loss = 0
    correct = 0
    total = 0
    getattr(tqdm, '_instances', {}).clear()  # ⬅ add this line
    loop = tqdm(train_loader)

    #for images, labels in train_loader:
    for images, labels in loop:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = loss_fn(outputs, labels, size_average=False)
        _, predicted = torch.max(outputs, dim=1)
        correct += (predicted == labels).sum().item()

        if half_precision:
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()

        optimizer.step()

        running_loss += loss.item()
        total += len(labels)

        loop.set_description('Epoch {}/{}'.format(epoch, epochs))
        loop.set_postfix(loss=loss.item())
        #print(".")

    loss = running_loss / total
    acc = correct / total * 100.

    getattr(tqdm, '_instances', {}).clear()  # ⬅ add this line
    #print(f"\nEpoch {epoch}/{epochs} acc: {acc:.4f} loss: {loss:.4f} | ", end = '')

    try:
        wandb.log({"Train Accuracy": acc, "Train Loss": loss })
    except:
        pass

    return acc, loss


def test(model, test_loader, epoch, epochs, loss_fn, device):
    model.eval()

    correct = 0
    total = 0
    running_loss = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels, size_average=False)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

    loss = running_loss / total
    acc = correct / total * 100.
    #print(f"Epoch {epoch}/{epochs} val_acc: {acc:.4f} val_loss: {loss:.4f}")
    getattr(tqdm, '_instances', {}).clear()  # ⬅ add this line
    print(f"\n**** val_acc: {acc:.4f} val_loss: {loss:.4f}")

    try:
        wandb.log({"val_acc": acc, "val_loss": loss })
    except:
        pass

    return acc, loss


def eval(model, test_loader, loss_fn, device):
    model.eval()

    correct = 0
    total = 0
    running_loss = 0
    
    example_images = []

    all_predictions = []
    ground_truth = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels, size_average=False)
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, dim=1)
            correct += (predicted == labels).sum().item()
            total += len(labels)

            all_predictions.extend([y_hat.item() for y_hat in predicted])
            ground_truth.extend([label.item() for label in labels])

            # Uploaded these images into W&B to view
            for image, prediction, label in zip(images, predicted, labels):
                prediction = idx_to_class[prediction.item()]
                label = idx_to_class[label.item()]
                example_images.append(wandb.Image(image, 
                    caption="Pred: {}\nLabel: {}".format(prediction, label)))
            
    # Calculate and print out information on accuracy and loss
    loss = running_loss / total
    acc = correct / total * 100.
    print(f"Accuracy: {acc:.4f} Loss: {loss:.4f}")

    try:
        wandb.log({"example_images": example_images, "eval_acc": acc, "eval_loss": loss })
    except:
        pass

    return acc, loss, ground_truth, all_predictions




def plot(history):
    train_acc, train_loss = history.train_acc, history.train_loss
    val_acc, val_loss = history.val_acc, history.val_loss

    plt.rcParams["figure.figsize"] = (10,4)
    plt.subplot(1, 2, 1)
    plt.plot(train_acc)
    plt.plot(val_acc)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    # summarize history for loss
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'dev'], loc='upper left')
    plt.tight_layout()
    plt.show()


def setup_wandb(project_name, model, batch_size, test_batch_size, num_epochs, learning_rate, momentum, log_interval):

    wandb.login()
    wandb.init(project = project_name)
    wandb.watch_called = False
    config = wandb.config
    config.batch_size = batch_size
    config.test_batch_size = batch_size
    config.epochs = num_epochs
    config.lr = learning_rate
    config.momentum = momentum
    config.log_interval = log_interval

    wandb.watch(model, log="all")


# Save model information
def save_model(directory, model, optimizer, history):
    epoch = len(history)
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history' : history
    }
    filename = os.path.join(directory, f'epoch_{epoch}.pt')
    #print("Saving model to ", filename)
    torch.save(state, filename)

def load_model(filepath, model, optimizer):

    print("Loading " + filepath)

    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    history = checkpoint['history']

    return epoch, history


# Change the learning rate ofr the optimizer
def set_optimizer_lr(optimizer, lr):
    for g in optimizer.param_groups:
        g['lr'] = lr



def get_class_mappings(train_data):
    class_to_idx = train_data.class_to_idx
    idx_to_class = dict()

    for category, idx in class_to_idx.items():
        idx_to_class[idx] = category

    return class_to_idx, idx_to_class


def load_best_checkpoint(ckpt_directory, model, optimizer, history):
    epoch = history.best_loss[0]

    filepath = os.path.join(ckpt_directory, f"epoch_{epoch}.pt")

    epoch, history = load_model(filepath, model, optimizer)

    return history

def load_last_checkpoint(ckpt_directory, model, optimizer, history):

    checkpoints = os.path.join(ckpt_directory, f"epoch_*.pt")
    checkpoints = glob.glob(checkpoints)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    filepath = checkpoints[0]

    epoch, history = load_model(filepath, model, optimizer)
    return history

   