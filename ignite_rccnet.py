from ignite_softmax_sccnn import softmaxSCCNN
import torch
import torch.nn as nn

class RCCnet(softmaxSCCNN):
    def __init__(self, in_channels=3, num_classes=4, dropout_p=0.2, loss_weights=None, criterion=None):
        super(RCCnet, self).__init__(in_channels=in_channels, num_classes=num_classes, dropout_p=dropout_p, loss_weights=loss_weights, criterion=criterion)


