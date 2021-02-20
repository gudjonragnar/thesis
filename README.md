# Classification of cells in soft oral tissue

This is my Masters thesis project. It is done in collaboration between KTH and the Department of Dental Science at Karolinska Institute. It is focused on classification of nuclei in soft oral tissue slides, especially lymphocytes (and other inflammatory cells). 


## Setup

This guide supposes you have Python 3.8 installed on your system.

1. `pip install virtualenv`
2. `virtualenv venv`
3. `source venv/bin/activate`
4. `pip install -r requirements.txt`
5. `pre-commit install`
6. (Optional) `echo "export KI_ROOT_DIR=<path/to/dir>" >> venv/bin/activate`
7. (Optional) `echo "export CRC_ROOT_DIR=<path/to/dir>" >> venv/bin/activate`

See the chapter on the two datasets for explanation of the variables.
Adding them to your venv activation script ensures that they are present after you source it, without adding it to your global environment.

:bulb: You have to run `source venv/bin/activate` again to load the variables.

The only current pre-commit hook is `black` formatting.

## Structure 

There are two main parts to this repository: `data` and `models`.
The `data` directory contains files related to splitting, creating and loading the datasets, while `models` contains the models and training parts.

### Creating the dataset
This work is created to support two datasets. The one that this work is centered around we call the KI dataset while the other one we call the CRC dataset.
The CRC dataset was used in the two main papers that this work is based on:
 
 1. [Locality Sensitive Deep Learning for Detection and Classification of Nuclei in Routine Colon Cancer Histology Images](https://ieeexplore.ieee.org/abstract/document/7399414)
 2. [RCCNet: An Efficient Convolutional Neural Network for Histological Routine Colon Cancer Nuclei Classification](https://ieeexplore.ieee.org/abstract/document/8581147)

The dataset can be downloaded from [here](https://warwick.ac.uk/fac/sci/dcs/research/tia/data/crchistolabelednucleihe/crchistophenotypes_2016_04_28.zip).

#### KI dataset
This is the main dataset used in this work. To prepare this dataset for use, you should set the environment variable `KI_ROOT_DIR` to point to the root directory of this dataset and the run `data/xml_parser.py`. The script will collect all centers and split the image into smaller images so loading the images during training is quicker.
This will create three `.npy` files in your directory: `class_weights.npy`, `train_list.npy`, and `test_list.npy`.

#### CRC dataset
The structure should be a directory containing all the image directories (img1, img2, ...) where each contains all information about said image.
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

To then create a train/test split set an environment variable `CRC_ROOT_DIR` to point to this directory (the one containing all the `imgN`) and run `data/combine_crc.py`.
This will create three `.npy` files in your directory: `class_weights.npy`, `train_list.npy`, and `test_list.npy`.

### Training
To train either the `SCCNN` or `RCCnet` you can use the `models/training.py` script. It has a CLI but you can also use the `train` function contained in the file. The CLI offers very basic choice of model architecture and optimizer. All the parameter choosing is done in `params.py`.
```
usage: training.py [-h] [-m {sccnn,rccnet}] [-o {adam,sgd}] [-d {ki,crc}]

Runs training for sccnet or rccnet

optional arguments:
  -h, --help            show this help message and exit
  -m {sccnn,rccnet}, --model {sccnn,rccnet}
                        Choose which model to use (default: None)
  -o {adam,sgd}, --optimizer {adam,sgd}
                        Choose which optimizer to use (default: OptimizerType.ADAM)
  -d {ki,crc}, --dataset {ki,crc}
                        Choose which dataset to use (default: DataSet.KI)}
```

The `VGG` transfer learning model is included in `models/vgg_transfer.py` which also contains the training script. 