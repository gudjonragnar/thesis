import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as models
import numpy as np
import os
import time
import copy
import params

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint
from torch.optim.lr_scheduler import MultiStepLR
from torchvision import transforms

from torch.utils.data import DataLoader
from data import ClassificationDataset

def stats(model, dataloader, metrics):
    for name, met in metrics.items():
        met.reset()
    for X, y in dataloader:
        y_pred = model(X)
        for name, met in metrics.items():
            met.update((y_pred,y))
    computed_metrics = {}
    for name, met in metrics.items():
        computed_metrics[name] = met.compute()
    return computed_metrics


def train_model(model, dataloaders, criterion, optimizer, metrics, num_epochs=25):
    since = time.time()

    val_F1_history = []
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_F1 = 0.0

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

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            computed_metrics = stats(model, dataloaders['val'], metrics)
            weights = loss._buffers['weight']
            F1 = (2*computed_metrics['precision']*computed_metrics['recall'] /
                (computed_metrics['precision']+computed_metrics['recall'])).numpy()
            F1[np.isnan(F1)] = 0
            weighted_F1 = np.sum(F1*weights.numpy())

            print('{} Loss: {:.4f}, Acc: {:.4f}, Weighted F1: {:.4f}'.format(phase, epoch_loss, computed_metrics['acc'], weighted_F1))

            # deep copy the model
            if phase == 'val' and weighted_F1 > best_F1:
                best_F1 = weighted_F1
                best_model_wts = copy.deepcopy(model.state_dict())
            if phase == 'val':
                val_F1_history.append(weighted_F1)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_F1))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_F1_history


# Freezes all layers if _feature_extracting_ is True
# This means that the layer parameters will not be updated during training
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



if __name__ == "__main__":
    feature_extract = True 
    num_epochs = 10
    input_size = 224
    train_transformations = transforms.Compose([
        transforms.Resize(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    test_transformations = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fetch model with pretrained weights
    model = models.vgg16(pretrained=True)
    set_parameter_requires_grad(model, feature_extract)

    # Replace top layer
    fc_in = model.classifier[-1].in_features
    new_layer = nn.Linear(in_features=fc_in, out_features=params.num_classes)
    model.classifier[-1] = new_layer

    # Move model to GPU if exists
    model = model.to(device)

    # Metrics to be calculated
    metrics = metrics={'accuracy': Accuracy(), 'precision': Precision(), 'recall': Recall()}

    criterion = nn.CrossEntropyLoss()

    # Collect parameters to be updated
    params_to_update = model.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)

    optimizer = torch.optim.SGD(params_to_update, lr=params.lr, momentum=params.momentum)

    # Datasets
    batch_size = 50
    root_dir = params.root_dir
    num_workers = params.num_workers
    train_ds = ClassificationDataset(root_dir=root_dir, train=True, shift=params.shift, transform=train_transformations)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_ds = ClassificationDataset(root_dir=root_dir, train=False, transform=test_transformations)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    dataloaders_dict = {'train':train_dl, 'val':test_dl}

    # Train
    model_ft, hist = train_model(model, dataloaders_dict, criterion, optimizer, metrics, num_epochs=num_epochs)
