from utils import EnumAction
import torch
import params
import numpy as np

from models.sccnn import SCCNN
from models.rccnet import RCCnet
from data.dataset import NEPValidationDataset, ClassificationDataset, DataSet
from ignite.engine import create_supervised_evaluator
from ignite.metrics import Accuracy, Precision, Recall
from torch.utils.data import DataLoader


def handler(output):
    pred, target = output
    pred = torch.sum(pred, dim=0) / len(target)
    stack = [pred] * len(target)
    return torch.stack(stack), target
    # test_output = torch.tensor(
    #     [[0.1, 0.1, 0.5, 0.3], [0.1, 0.1, 0.5, 0.3], [0.1, 0.1, 0.5, 0.3]]
    # )
    # test_target = torch.tensor([1, 1, 1])

    # print(handler((test_output,test_target)))


def evaluate(metrics):
    F1 = (
        2 * metrics["prec"] * metrics["rec"] / (metrics["prec"] + metrics["rec"])
    ).numpy()
    F1[np.isnan(F1)] = 0
    print(
        "Single patch:\n\taccuracy: {}\n\tprecision: {}\n\trecall: {}\n\tF1: {}".format(
            metrics["acc"], metrics["prec"], metrics["rec"], F1
        )
    )


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Runs training for sccnet or rccnet")
    parser.add_argument(
        "-m",
        "--model",
        dest="model",
        choices=["sccnn", "rccnet"],
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=DataSet,
        action=EnumAction,
        default=DataSet.CLASSIFICATION,
    )
    parser.add_argument(
        "-p",
        "--model-path",
        dest="path",
        type=str,
    )

    args = parser.parse_args()

    if args.model == "sccnn":
        model = SCCNN()
        size = 27
    elif args.model == "rccnet":
        model = RCCnet()
        size = 32
    else:
        raise Exception("Chosen model not supported")

    if args.dataset == DataSet.CLASSIFICATION:
        evaluator = create_supervised_evaluator(
            model, metrics={"acc": Accuracy(), "rec": Recall(), "prec": Precision()}
        )
        test_ds = ClassificationDataset(
            root_dir=params.root_dir, width=size, height=size, train=False
        )
        test_dl = DataLoader(
            test_ds, batch_size=params.batch_size, num_workers=params.num_workers
        )
    elif args.dataset == DataSet.NEP:
        evaluator = create_supervised_evaluator(
            model,
            metrics={
                "acc": Accuracy(handler),
                "rec": Recall(handler),
                "prec": Precision(handler),
            },
        )
        test_ds = NEPValidationDataset(root_dir=params.root_dir, d=4)
        test_dl = DataLoader(
            test_ds,
            batch_size=len(test_ds.shifts),
            num_workers=params.num_workers,
            drop_last=True,
        )
    else:
        raise Exception("Chosen dataset not available")

    print(f"Chosen model: {args.model}, chosen dataset: {args.dataset.value}")

    evaluator.run(test_dl)
    metrics = evaluator.state.metrics
    evaluate(metrics)

    print("Loading trained model!")
    model.load_model(args.path)
    print("Loading finished!")

    evaluator.run(test_dl)
    metrics = evaluator.state.metrics
    evaluate(metrics)
