from enum import Enum
from utils import EnumAction
import os
import numpy as np
from typing import Union
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import time
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR

from data.dataset import ClassificationDataset
from models.rccnet import RCCnet
from models.sccnn import SCCNN
from params import Params, sccnn_params, rccnet_params


Model = Union[SCCNN, RCCnet]


class OptimizerType(Enum):
    ADAM = "adam"
    SGD = "sgd"


def train(model_class: Model, params: Params, optim_type: OptimizerType):
    num_classes = params.num_classes
    batch_size = params.batch_size
    num_workers = params.num_workers
    root_dir = params.root_dir
    if model_class == SCCNN:
        width = height = 27
    elif model_class == RCCnet:
        width = height = 32
    else:
        raise Exception(
            "Model_class should be either a softmax SCCNN or RCCnet class initializer"
        )
    train_ds = ClassificationDataset(root_dir=root_dir, train=True, shift=params.shift)
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    test_ds = ClassificationDataset(
        root_dir=root_dir, train=False, width=width, height=height
    )
    test_dl = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)

    class_weight_dict = np.load(
        os.path.join(root_dir, "class_weights.npy"), allow_pickle=True
    ).item()
    class_weights = torch.tensor(
        [class_weight_dict[i] / class_weight_dict["total"] for i in range(num_classes)],
        dtype=torch.float,
    )
    model: Model = model_class(
        loss_weights=class_weights, num_classes=num_classes, dropout_p=params.dropout_p
    )

    if optim_type == OptimizerType.ADAM:
        optimizer = torch.optim.Adam(model.parameters(), lr=params.lr)
        scheduler = None
    elif optim_type == OptimizerType.SGD:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay,
            momentum=params.momentum,
        )
        step_scheduler = StepLR(optimizer, step_size=params.lr_step_size, gamma=0.1)
        scheduler = LRScheduler(step_scheduler)
    else:
        raise Exception("Unsupported optimizer type")

    criterion = nn.NLLLoss(weight=model.class_weights)

    time1 = time.time()
    model.train_model(
        train_dl,
        optimizer,
        criterion,
        max_epochs=params.epochs,
        val_loader=test_dl,
        scheduler=scheduler,
    )
    time2 = time.time()
    print(
        "It took {:.5f} seconds to train {} epochs, average of {:.5f} sec/epoch".format(
            (time2 - time1), params.epochs, (time2 - time1) / params.epochs
        )
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Runs training for sccnet or rccnet")
    parser.add_argument(
        "-n",
        "--network",
        dest="network",
        choices=["sccnn", "rccnet"],
    )
    parser.add_argument(
        "-o",
        "--optimizer",
        dest="optim",
        type=OptimizerType,
        action=EnumAction,
        default=OptimizerType.ADAM,
    )

    args = parser.parse_args()

    if args.network == "sccnn":
        train(SCCNN, sccnn_params, args.optim)
    elif args.network == "rccnet":
        train(RCCnet, rccnet_params, args.optim)
