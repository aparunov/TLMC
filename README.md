# Usage guide

This repository was created as part of my Master thesis: Transfer learning of interatomic potentials: from molecules to crystals. 

The aim was to use ANI2x model and apply it to crystals as a part of the Seventh CCDC Blind Test of Crystal
Structure Prediction Methods.  

## HARDWER REQUIREMENTS

NVIDIA A100 40GB

## Installation steps 

### Run docker container 

docker run  --gpus all   -ti --ipc=host  -v /PATH/TO/FOLDER:/app nvidia/cuda:11.2.0-devel-ubuntu20.04

### Install pip 

apt update install python3-pip

### Install pytorch with CUDA support 

pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html

### Install wget 

apt install wget

### Install conda 

mkdir -p ~/miniconda3  && \
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh && \
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 && \
rm -rf ~/miniconda3/miniconda.sh
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh

### Install torchani with CUDA support

conda install -c conda-forge nnpops

### Install additional dependencies

pip install h5py ase 

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



