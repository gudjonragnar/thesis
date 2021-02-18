# Classification of cells in soft oral tissue

This is my Masters thesis project. It is done in collaboration between KTH and the Department of Dental Science at Karolinska Institute. It is focused on classification of nuclei in soft oral tissue slides, especially lymphocytes (and other inflammatory cells). 


## Setup

This guide supposes you have Python 3.8 installed on your system.

1. `pip install virtualenv`
2. `virtualenv venv`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`
5. `pre-commit install`

The only current pre-commit hook is `black` formatting.

## Structure 

There are two main parts to this repository: `data` and `models`.
The `data` directory contains files related to splitting, creating and loading the datasets, while `models` contains the models and training parts.

### Creating the dataset
The data itself should be located on your system. The structure should be a directory containing all the image directories (img1, img2, ...) where each contains all information about said image.
This is important, since the code is setup to read this structure.

Here is an example of how the image directory tree should look like.
```
├── img1
│   ├── img1.bmp
│   ├── img1_epithelial.mat
│   ├── img1_fibroblast.mat
│   ├── img1_inflammatory.mat
│   ├── img1.npy
│   └── img1_others.mat
```

To then create a train/test split set the root path in `params.py` to point to this directory (the one containing all the `imgN`) and run `data/combine_all.py`.
This will create three `.npy` files in your directory: `class_weights.npy`, `train_list.npy`, and `test_list.npy`.

### Training
To train either the `SCCNN` or `RCCnet` you can use the `models/training.py` script. It has a CLI if but you can also use the `train` function contained in the file. The CLI offers very basic choice of model architecture and optimizer. All the parameter choosing is done in `params.py`.
```
usage: training.py [-h] [-n {sccnn,rccnet}] [-o {adam,sgd}]

Runs training for sccnet or rccnet

optional arguments:
  -h, --help            show this help message and exit
  -n {sccnn,rccnet}, --network {sccnn,rccnet}
  -o {adam,sgd}, --optimizer {adam,sgd}
```

The `VGG` transfer learning model is included in `models/vgg_transfer.py` which also contains the training script. 