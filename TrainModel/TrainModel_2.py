# Train model following the official document

from __future__ import print_function
from __future__ import division

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from models import return_pytorch04_xception, return_pytorch04_xception_ft
from transform import xception_default_data_transforms

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# Top level data directory. Here we assume the format of the directory conforms
#   to the ImageFolder structure
# data_dir = "/home/jc/Faceforensics_onServer/Final_Faceforensics++"
data_dir = "/home/jc/Faceforensics_onServer/Final_Faceforensics++no_NT-Big"  # no NT

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "squeezenet"

# Number of classes in the dataset(to fine-tune)
num_classes = 2

# Batch size for training (change depending on how much memory you have)
batch_size = 32


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


# if we are feature extracting and only want to compute gradients for the newly initialized layer
# then we want all of the other parameters to not require gradients. This will make more sense later.
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        # for param in model.parameters():
        #     param.requires_grad = False
        for para in list(model.parameters())[:-1]:
            para.requires_grad = False


if __name__ == '__main__':
    print("Initializing Datasets and Dataloaders...")

    # Create training and validation datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), xception_default_data_transforms[x])
                      for x in ['train', 'val']}
    # Create training and validation dataloaders
    dataloaders_dict = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=0)
        for x in ['train', 'val']}

    # Detect if we have a GPU available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    p = parser = argparse.ArgumentParser(description='Process some integers.')
    p.add_argument('--pretrain3epochs', action='store_true')
    args = p.parse_args()

    if args.pretrain3epochs:
        pretrain3epoch = True
        num_epochs = 0
        print('True')
    else:
        pretrain3epoch = False
        num_epochs = 15
        print('False')


    model_ft = return_pytorch04_xception_ft(True, pretrain3epoch)
    set_parameter_requires_grad(model_ft, pretrain3epoch)
    if pretrain3epoch:
        num_ftrs = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_ftrs, num_classes)

    print(model_ft)

    model_ft = model_ft.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if pretrain3epoch:
        params_to_update = []
        for name, param in model_ft.named_parameters():

            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)


    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    # Train and evaluate
    model_ft, hist = train_model(model_ft, dataloaders_dict, criterion, optimizer_ft, num_epochs=num_epochs)

    if pretrain3epoch:
        # torch.save(model_ft.state_dict(), '/home/jc/Faceforensics_onServer/Model/xception-b5690688-after3epochs-noNT-Big.pth')
        torch.save(model_ft.state_dict(),
                   '/home/jc/Faceforensics_onServer/Model/xception-b5690688-after0epochs-noNT-Big.pth')
    else:
        torch.save(model_ft.state_dict(),
                   '/home/jc/Faceforensics_onServer/Model/xception-b5690688-after15epochs-noNT-Big.pth')
