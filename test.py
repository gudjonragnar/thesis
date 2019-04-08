
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

from ignite_softmax_sccnn import softmaxSCCNN

from torch.utils.data import DataLoader
from data import ClassificationDataset



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

    test_dl_iter = iter(test_dl)
    X, label = next(test_dl_iter)

    out = model.forward(X)

    