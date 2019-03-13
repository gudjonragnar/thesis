import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os
import time
import params

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint
# from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import MultiStepLR

from torch.utils.data import DataLoader
from data import ClassificationDataset

class softmaxSCCNN(nn.Module):
    """
    A Convolutional neural network with a softmax classifier SC-CNN implemented in Sirunakunwattana et al. 
    """
    def __init__(self, in_channels=3, num_classes=4, dropout_p=0.2, loss_weights=None, criterion=None):
        super(softmaxSCCNN, self).__init__()
        # Globals
        self.num_classes = num_classes
        self.p = dropout_p
        self.class_weights = loss_weights
        self.model_name = 'sccnn'


        # Layers
        self.c1 = nn.Conv2d(3, 36, (4,4), stride=1)
        self.c2 = nn.Conv2d(36, 48, (3,3), stride=1)
        self.fc1 = nn.Linear(5*5*48,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,self.num_classes)

    def forward(self, X):
        """
        Implements the forward pass of the SC-CNN network
        """
        y = F.relu(self.c1(X))
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(self.c2(y))
        y = F.max_pool2d(y, 2, 2)
        y = y.view(-1, 5*5*48)
        y = F.relu(self.fc1(y))
        y = F.dropout(y, p=self.p)
        y = F.relu(self.fc2(y))
        y = F.dropout(y, p=self.p)
        y = F.log_softmax(self.fc3(y), dim=1)
        
        # y = self.fc3(y)

        return y

    def evaluate(self, data, target):
        """
        Evaluates the network on _data_ and returns the percentage of correct classifications 
        """
        self.eval()
        out = F.softmax(self.forward(data), dim=1)
        out_choices = torch.argmax(out, 1)
        correct_percentage = torch.sum(torch.where(target==out_choices,torch.tensor(1.),torch.tensor(0.)))/len(target)
        return correct_percentage
    
    # This function is to evaluate using the Neighbouring Ensamble Predictor (NEP)
    # The input _nep_dataset_ will output a stack of predictions, one for each center-shift, along with the target
    # We will wrap this with a dataloader with batch_size=1 and then combine the predictions for each batch. Finally we will compare
    # def evaluate_NEP(self, nep_dataset):

    def save_model(self, epoch, all=False):
        """
        Saves the model. If _all_ is true then it will save the whole model, otherwise only its parameters.
        """
        filename = 'checkpoints/{}_model_{}.pth'.format(self.model_name, epoch)
        if all:
            torch.save(self, filename)
        else:
            torch.save(self.state_dict(), filename)

    def load_model(self, filename, all=False):
        """
        Loads a model. If _all_ is true then it will load a whole model, otherwise only its parameters.
        Puts the model into evaluation mode.
        """
        if all:
            self = torch.load(filename.format('model'))
        else:
            self.load_state_dict(torch.load(filename))
        self.eval()

    def train_model(self, train_loader, optimizer, criterion, max_epochs, val_loader=None, init=True):
        """
        This method trains the model for _num_epochs_.
        _train_loader_ should be a DataLoader returning a tuple of (data, target).
        """
        if init:
            self.apply(init_weights)
        trainer = create_supervised_trainer(self, optimizer, criterion)
        evaluator = create_supervised_evaluator(self, 
            metrics={'accuracy': Accuracy(), 'precision': Precision(), 'recall': Recall(), 'loss': Loss(criterion)})

        # step_scheduler = MultiStepLR(optimizer, milestones=[60, 100], gamma=0.1)
        # scheduler = LRScheduler(step_scheduler)

        # Print out metrics with some defined interval (if statement)
        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(trainer):
            if trainer.state.epoch % 1 == 0:
                evaluator.run(val_loader)
                metrics = evaluator.state.metrics
                F1 = (2*metrics['precision']*metrics['recall']/(metrics['precision']+metrics['recall'])).numpy()
                F1[np.isnan(F1)] = 0
                print(
                    ("After {} epochs, Accuracy = {:.4f}, Loss = {:.6f}\n\tPrec\t={}\n\tRecall\t={}\n\t"+ \
                        "F1\t={}\n\tWeighted Average F1 score: {}\n")
                    .format(trainer.state.epoch, 
                        metrics['accuracy'], 
                        metrics['loss'], 
                        metrics['precision'].numpy(), 
                        metrics['recall'].numpy(),
                        F1,
                        np.sum(F1*self.class_weights.numpy())))
        
        @trainer.on(Events.EPOCH_COMPLETED)
        def saver(trainer):
            if trainer.state.epoch % params.save_interval == 0:
                self.save_model(trainer.state.epoch)

        # Instead of the LR scheduler, since it was not possible to successfully
        # install from github on tcs111
        @trainer.on(Events.EPOCH_COMPLETED)
        def anneal_lr(trainer):
            if trainer.state.epoch in params.lr_decay_epochs:
                change_lr(optimizer)
        
        checkpointer = ModelCheckpoint('checkpoints', 'sccnn', save_interval=5, create_dir=True, require_empty=False)
        # trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': self})
        # trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

        print("Start training!")
        trainer.run(train_loader, max_epochs=max_epochs)
        print("Training finished")


def change_lr(optim, gamma=0.1):
    for g in optim.param_groups:
        g['lr'] *= gamma

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight, std=0.02)
        m.bias.data.fill_(0.0)

def init_weights_linear(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.02)
        m.bias.data.fill_(0.0)

if __name__ == "__main__":
    num_classes = params.num_classes
    batch_size = params.batch_size
    num_workers = params.num_workers
    root_dir = params.root_dir
    train_ds = ClassificationDataset(root_dir=root_dir, train=True, shift=params.shift)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_ds = ClassificationDataset(root_dir=root_dir, train=False)
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    class_weight_dict = np.load(os.path.join(root_dir,'class_weights.npy')).item()
    class_weights = torch.tensor([class_weight_dict[i]/class_weight_dict['total'] for i in range(num_classes)], dtype=torch.float)
    model = softmaxSCCNN(loss_weights=class_weights, num_classes=num_classes)

    # data = torch.rand(3,3,27,27)

    # optimizer = torch.optim.Adam(net.parameters(), lr=10)
    # optimizer = torch.optim.Adam(net.parameters(), lr=1, weight_decay=5e-4)
    optimizer = torch.optim.SGD(model.parameters(), 
        lr=params.lr, 
        weight_decay=params.weight_decay,
        momentum=params.momentum)
    
    criterion = nn.NLLLoss(weight=model.class_weights)

    def t(n=3):
        time1 = time.time()
        model.train_model(train_dl, optimizer, criterion, max_epochs=n, val_loader=test_dl)
        # net.train_model(train_dl, optimizer, num_epochs=n, val_loader=None)
        time2 = time.time()
        print("It took {:.5f} seconds to train {} epochs, average of {:.5f} sec/epoch".format((time2-time1), n, (time2-time1)/n))

    # Uncomment for training
    t(params.epochs)




