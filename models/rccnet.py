from models.sccnn import SCCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from params import rccnet_params as params
import numpy as np
import os
import time

from torch.utils.data import DataLoader
from data.dataset import ClassificationDataset


class RCCnet(SCCNN):
    def __init__(
        self,
        num_classes: int = 4,
        dropout_p: float = 0.2,
        batch_norm: bool = False,
        loss_weights=None,
    ):
        super(RCCnet, self).__init__(
            num_classes=num_classes,
            dropout_p=dropout_p,
            loss_weights=loss_weights,
        )
        self.model_name = "rccnet"
        self.batch_norm = batch_norm

        # Layers
        self.c1 = nn.Conv2d(3, 32, (3, 3), stride=1, padding=1)
        self.c2 = nn.Conv2d(32, 32, (3, 3), stride=1)
        self.c3 = nn.Conv2d(32, 64, (3, 3), stride=1, padding=1)
        self.c4 = nn.Conv2d(64, 64, (3, 3), stride=1)
        self.fc1 = nn.Linear(6 * 6 * 64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.num_classes)

        # Batch Norm
        self.bn1 = nn.BatchNorm2d(num_features=32)
        self.bn2 = nn.BatchNorm2d(num_features=64)
        self.bn3 = nn.BatchNorm1d(num_features=512)

    def forward(self, X):
        y = F.relu(self.c1(X))
        y = self.bn1(y) if self.batch_norm else y
        y = F.relu(self.c2(y))
        y = self.bn1(y) if self.batch_norm else y
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(self.c3(y))
        y = self.bn2(y) if self.batch_norm else y
        y = F.relu(self.c4(y))
        y = self.bn2(y) if self.batch_norm else y
        y = F.max_pool2d(y, 2, 2)
        y = y.view(-1, 6 * 6 * 64)
        y = F.relu(self.fc1(y))
        y = F.dropout(y, p=self.p)
        y = self.bn3(y) if self.batch_norm else y
        y = F.relu(self.fc2(y))
        y = F.dropout(y, p=self.p)
        y = self.bn3(y) if self.batch_norm else y
        y = F.log_softmax(self.fc3(y), dim=1)

        return y
