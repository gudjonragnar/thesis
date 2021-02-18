import torch
import torch.cuda
import torch.nn as nn
import torch.optim
import torch.nn.functional as F
import numpy as np
from params import sccnn_params as params


from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, Precision, Recall
from ignite.handlers import ModelCheckpoint


class SCCNN(nn.Module):
    """
    A Convolutional neural network with a softmax classifier SC-CNN implemented in Sirunakunwattana et al.
    """

    def __init__(
        self,
        num_classes=4,
        dropout_p=0.2,
        loss_weights=None,
    ):
        super(SCCNN, self).__init__()
        # Globals
        self.num_classes = num_classes
        self.p = dropout_p
        self.class_weights = loss_weights
        self.model_name = "sccnn"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_weights = self.class_weights.to(self.device)

        # Layers
        self.c1 = nn.Conv2d(3, 36, (4, 4), stride=1)
        self.c2 = nn.Conv2d(36, 48, (3, 3), stride=1)
        self.fc1 = nn.Linear(5 * 5 * 48, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.num_classes)

    def forward(self, X):
        """
        Implements the forward pass of the SC-CNN network
        """
        y = F.relu(self.c1(X))
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(self.c2(y))
        y = F.max_pool2d(y, 2, 2)
        y = y.view(-1, 5 * 5 * 48)
        y = F.relu(self.fc1(y))
        y = F.dropout(y, p=self.p)
        y = F.relu(self.fc2(y))
        y = F.dropout(y, p=self.p)
        y = F.log_softmax(self.fc3(y), dim=1)

        return y

    def evaluate(self, data, target):
        """
        Evaluates the network on _data_ and returns the percentage of correct classifications
        """
        self.eval()
        out = F.softmax(self.forward(data), dim=1)
        out_choices = torch.argmax(out, 1)
        correct_percentage = torch.sum(
            torch.where(target == out_choices, torch.tensor(1.0), torch.tensor(0.0))
        ) / len(target)
        return correct_percentage

    def save_model(self, epoch, all=False):
        """
        Saves the model. If _all_ is true then it will save the whole model, otherwise only its parameters.
        """
        filename = "checkpoints/{}_model_{}.pth".format(self.model_name, epoch)
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
            self = torch.load(filename.format("model"))
        else:
            self.load_state_dict(torch.load(filename))
        self.eval()

    def train_model(
        self,
        train_loader,
        optimizer,
        criterion,
        max_epochs,
        val_loader=None,
        init=True,
        scheduler=None,
    ):
        """
        This method trains the model for _num_epochs_.
        _train_loader_ should be a DataLoader returning a tuple of (data, target).
        """
        if init:
            self.apply(init_weights)
        trainer = create_supervised_trainer(
            self, optimizer, criterion, device=self.device
        )
        evaluator = create_supervised_evaluator(
            self,
            device=self.device,
            metrics={
                "accuracy": Accuracy(),
                "precision": Precision(),
                "recall": Recall(),
                "loss": Loss(criterion),
            },
        )
        if scheduler:
            trainer.add_event_handler(Events.ITERATION_COMPLETED, scheduler)

        # Print out metrics with some defined interval
        @trainer.on(Events.EPOCH_COMPLETED(every=params.eval_interval))
        def validate(trainer):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            F1 = (
                (
                    2
                    * metrics["precision"]
                    * metrics["recall"]
                    / (metrics["precision"] + metrics["recall"])
                )
                .cpu()
                .numpy()
            )
            F1[np.isnan(F1)] = 0
            print(
                (
                    "After {} epochs, Accuracy = {:.4f}, Loss = {:.6f}\n\tPrec\t={}\n\tRecall\t={}\n\t"
                    + "F1\t={}\n\tWeighted Average F1 score: {}\n"
                ).format(
                    trainer.state.epoch,
                    metrics["accuracy"],
                    metrics["loss"],
                    metrics["precision"].cpu().numpy(),
                    metrics["recall"].cpu().numpy(),
                    F1,
                    np.sum(F1 * self.class_weights.cpu().numpy()),
                )
            )

        checkpointer = ModelCheckpoint(
            "checkpoints",
            "sccnn",
            create_dir=True,
            require_empty=False,
        )
        trainer.add_event_handler(
            Events.EPOCH_COMPLETED(every=params.save_interval),
            checkpointer,
            {"model": self},
        )

        print("Start training!")
        trainer.run(train_loader, max_epochs=max_epochs)
        print("Training finished")


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.normal_(m.weight, std=0.02)
        m.bias.data.fill_(0.0)


def init_weights_linear(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.02)
        m.bias.data.fill_(0.0)
