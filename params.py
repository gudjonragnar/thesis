from dataclasses import dataclass

root_dir = "/home/gudjon/CRCHistoPhenotypes_2016_04_28/Classification"


@dataclass
class Params:
    dropout_p: float
    num_classes: int
    num_workers: int
    lr: float
    weight_decay: float
    momentum: float
    batch_size: int
    epochs: int
    multiples: int
    eval_interval: int
    save_interval: int
    shift: int
    lr_step_size: int
    root_dir: str = root_dir


sccnn_params = Params(
    dropout_p=0.2,
    num_classes=4,
    num_workers=8,
    lr=0.001,
    weight_decay=5e-4,
    momentum=0.9,
    batch_size=100,
    epochs=10,
    multiples=1,
    eval_interval=1,
    save_interval=10,
    shift=3,
    lr_step_size=10,
)

rccnet_params = Params(
    dropout_p=0.5,
    num_classes=4,
    num_workers=8,
    lr=6e-5,
    weight_decay=5e-4,
    momentum=0.9,
    batch_size=100,
    epochs=500,
    multiples=10,
    save_interval=50,
    eval_interval=20,
    shift=6,
    lr_step_size=10,
)