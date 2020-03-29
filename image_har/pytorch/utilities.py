#!/usr/bin/env

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import wandb
from apex import amp

def train(model, train_loader, epoch, epochs, optimizer, loss_fn, device, train_acc=[], train_loss=[], half_precision=False):
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
    print(f"Epoch {epoch}/{epochs} acc: {acc:.4f} loss: {loss:.4f} | ", end = '')

    train_loss.append(loss)
    train_acc.append(acc)

    try:
        wandb.log({"Train Accuracy": acc, "Train Loss": loss })
    except:
        pass

    return train_acc, train_loss


def test(model, test_loader, epoch, epochs, loss_fn, device, val_acc=[], val_loss=[]):
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
    val_loss.append(loss)
    val_acc.append(acc)
    #print(f"Epoch {epoch}/{epochs} val_acc: {acc:.4f} val_loss: {loss:.4f}")
    print(f" val_acc: {acc:.4f} val_loss: {loss:.4f}")

    try:
        wandb.log({"val_acc": acc, "val_loss": loss })
    except:
        pass

    return val_acc, val_loss


def eval(model, test_loader, idx_to_class, loss_fn, device):
    model.eval()

    correct = 0
    total = 0
    running_loss = 0
    
    example_images = []
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

            # Uploaded these images into W&B to view
            for image, prediction, label in zip(images, predicted, labels):
                prediction = idx_to_class[prediction.item()]
                label = idx_to_class[label.item()]
                example_images.append(wandb.Image(image, 
                    caption="Pred: {}\nLabel: {}".format(prediction, label)))
            
  
    loss = running_loss / total
    acc = correct / total * 100.
    print(f"Accuracy: {acc:.4f} Loss: {loss:.4f}")

    try:
        wandb.log({"example_images": example_images, "eval_acc": acc, "eval_loss": loss })
    except:
        pass

    return acc, loss



def plot(train_acc, train_loss, val_acc, val_loss):
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
def save_model(filepath, epoch, model, optimizer, history):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'history' : history
    }
    torch.save(state, filepath)

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