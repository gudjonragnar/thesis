import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
import os

from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers.param_scheduler import LRScheduler
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
        if not criterion:
            self.criterion = nn.CrossEntropyLoss(weight=loss_weights)


        # Layers
        self.c1 = nn.Conv2d(3, 36, (4,4), stride=1)
        self.c2 = nn.Conv2d(36, 48, (3,3), stride=1)
        self.fc1 = nn.Linear(5*5*48,512)
        self.fc2 = nn.Linear(512,512)
        self.fc3 = nn.Linear(512,self.num_classes)


    def save_model(self, filename='model/{}.pth', all=False):
        """
        Saves the model. If _all_ is true then it will save the whole model, otherwise only its parameters.
        """
        if all:
            torch.save(self, filename.format('model'))
        else:
            torch.save(self.state_dict(), filename.format('statedict'))

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

    def save_checkpoint(self, epoch, optimizer, criterion ,filename='checkpoints/model_{}.tar'):
        """
        Saves the model along with the current training parameters for further training.
        Good to save regularly during training in case of failure or if manual tweaking is wanted.
        """
        from datetime import datetime

        current_datetime = datetime.now().strftime('%Y-%m-%d_%H_%M')
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss_criterion': criterion
            }, filename.format(current_datetime))

    def load_checkpoint(self, filename):
        """
        Returns the loaded checkpoint dictionary detailed in the save_checkpoint method
        and sets the model parameters and loss criterion to those of the checkpoint.
        The return value can be used to continue training, as inputs to the train_model method.
        """
        checkpoint_dict = torch.load(filename)
        self.load_state_dict(checkpoint_dict['model_state_dict'])
        self.criterion = checkpoint_dict['loss_criterion']
        return checkpoint_dict


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
        # y = F.softmax(self.fc3(y), dim=1)
        y = self.fc3(y)

        return y

    def evaluate(self, data, target):
        """
        Evaluates the network on _data_ and returns the percentage of correct classifications 
        """
        self.eval()
        out = F.softmax(self.forward(data), dim=1)
        out_choices = torch.argmax(out, 1)
        correct_percentage = torch.sum(torch.where(target==out_choices,torch.tensor(1.),torch.tensor(0.)))/len(target)
        if np.random.rand() < 0.05 and False:
            print(target)
            print(out_choices)
        return correct_percentage
    

    def train_model(self, train_loader, optimizer, max_epochs, val_loader=None, init=True):
        """
        This method trains the model for _num_epochs_.
        _train_loader_ should be a DataLoader returning a tuple of (data, target).
        """
        if init:
            self.apply(init_weights)
        trainer = create_supervised_trainer(self, optimizer, self.criterion)
        evaluator = create_supervised_evaluator(self, metrics={'accuracy': Accuracy(), 'loss':Loss(self.criterion)})

        step_scheduler = MultiStepLR(optimizer, milestones=[60, 100], gamma=0.1)
        scheduler = LRScheduler(step_scheduler)

        @trainer.on(Events.EPOCH_COMPLETED)
        def validate(trainer):
            if trainer.state.iteration % 1 == 0:
                evaluator.run(val_loader)
                metrics = evaluator.state.metrics
                print("After {} epochs, accuracy = {:.4f}, loss = {:.6f}".format(trainer.state.epoch, metrics['accuracy'], metrics['loss']))
        
        checkpointer = ModelCheckpoint('checkpoints', 'ignite', save_interval=10, create_dir=True, require_empty=False)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': self})
        # trainer.add_event_handler(Events.EPOCH_STARTED, scheduler)

        print("Start training!")
        trainer.run(train_loader, max_epochs=max_epochs)
        print("Training finished")


def change_lr(optim, new_lr):
    for g in optim.param_groups:
        g['lr'] = new_lr

def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight, std=0.01)
        m.bias.data.fill_(0.0)

def init_weights_linear(m):
    if type(m) == nn.Linear:
        torch.nn.init.normal_(m.weight, std=0.02)
        m.bias.data.fill_(0.0)


    
if __name__ == "__main__":
    root_dir = '/Users/gudjonragnar/Documents/KTH/Thesis/CRCHistoPhenotypes_2016_04_28/Classification'
    train_ds = ClassificationDataset(root_dir=root_dir, train=True, shift=4.)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=4)

    test_ds = ClassificationDataset(root_dir=root_dir, train=False)
    test_dl = DataLoader(test_ds, batch_size=32, num_workers=4)

    class_weight_dict = np.load(os.path.join(root_dir,'class_weights.npy')).item()
    class_weights = torch.tensor([class_weight_dict[i]/class_weight_dict['total'] for i in range(4)], dtype=torch.float)
    net = softmaxSCCNN(loss_weights=class_weights)

    # data = torch.rand(3,3,27,27)

    # optimizer = torch.optim.Adam(net.parameters(), lr=0.1, weight_decay=5e-4)
    # optimizer = torch.optim.Adam(net.parameters(), lr=1, weight_decay=5e-4)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01, weight_decay=5e-4, momentum=0.9)


    def t(n=3):
        net.train_model(train_dl, optimizer, max_epochs=n, val_loader=test_dl)
        # net.train_model(train_dl, optimizer, num_epochs=n, val_loader=None)

    t(120)


