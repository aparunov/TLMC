# Usage guide

This repository was created as part of my Master thesis: Transfer learning of interatomic potentials: from molecules to crystals. 

The aim was to use ANI2x model and apply it to crystals as a part of the Seventh CCDC Blind Test of Crystal
Structure Prediction Methods.  

## HARDWER REQUIREMENTS

Cuda 11.1 compatible graphics card

## Installation steps 

### Run docker container 

docker run  --gpus all   -ti --ipc=host  -v /PATH/TO/FOLDER:/app nvidia/cuda:11.2.0-devel-ubuntu20.04

### Install pip 

apt update install python3-pip

### Install pytorch with CUDA support 

pip3 install torch==1.9.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html tqdm ase torchani seaborn 

## Training

Datasets are assumed to be in the num.traj format in folder data, where num represents the structure number. 

### Convert num.traj to trainable format

python3 generate_dataset.py -n num.traj

Takes the num.traj file and creates the /data/train_data_num.pkl file which can be fed directlly to model.

### Basic model training 

python3 train.py -n num

model is saved at model/compiled_model_num.pt directory.

### Hyperparamter selection

python3 train.py --help

## Testing

### Test

python3 test.py -n num

## Utils

# Change model device to cpu

python3 utils/model2cpu.py -n num



