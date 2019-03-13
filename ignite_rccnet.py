from ignite_softmax_sccnn import softmaxSCCNN
import torch
import torch.nn as nn
import torch.nn.functional as F
import params

class RCCnet(softmaxSCCNN):
    def __init__(self, in_channels=3, num_classes=4, dropout_p=0.2, loss_weights=None):
        super(RCCnet, self).__init__(in_channels=in_channels, num_classes=num_classes, dropout_p=params.dropout_p, loss_weights=loss_weights)

        # Layers
        self.c1 = nn.Conv2d(3, 32, (3,3), stride=1, padding=1)
        self.c2 = nn.Conv2d(32, 32, (3,3), stride=1)
        self.c3 = nn.Conv2d(32, 64, (3,3), stride=1, padding=1)
        self.c4 = nn.Conv2d(64, 64, (3,3), stride=1)
        self.fc1 = nn.Linear(6*6*64, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, self.num_classes)

    def forward(self, X):
        y = F.relu(self.c1(X))
        y = F.relu(self.c2(y))
        y = F.max_pool2d(y, 2, 2)
        y = F.relu(self.c3(y))
        y = F.relu(self.c4(y))
        y = F.max_pool2d(y, 2, 2)
        y = y.view(-1, 6*6*64)
        y = F.relu(self.fc1(y))
        y = F.dropout(y, p=self.p)
        y = F.relu(self.fc2(y))
        y = F.dropout(y, p=self.p)
        y = F.log_softmax(self.fc3(y), dim=1)

        return y