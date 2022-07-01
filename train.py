import os
import argparse
import math

import numpy as np
import torch
import torchani
import torchani.data
import tqdm
from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
from ase.units import Hartree
from tqdm import tqdm
import pickle
import copy

from models import Model

parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', type=str, required=True, help='Model name')
parser.add_argument('-tl',
                    '--transfer-learning', 
                    type=int, 
                    default = 0, 
                    required=False, 
                    help=
                    'If set to 0 all layers are unfrozen and weights are taken from ANI2x model. 
                    If set to 1, top and bottom layer will be frozen. 
                    If set to 2 no transfer learning takes place and weight are drawn from the normal distribution'
                   )

parser.add_argument('-b',
                    '--batch-size', 
                    type=int, default=2, 
                    required=False, 
                    help = 'Set batch size.')

parser.add_argument('-me',
                    '--max-epochs', 
                    type=int, default=200, 
                    required=False, 
                    help = 'Maximum numbr of epochs')

parser.add_argument('-fc',
                    '--force-loss-coefficient', 
                    type=float, 
                    default = 0.9, 
                    required=False, 
                    help = 
                    'Force loss contribution to total loss; fc = fl/(fl+el) ')

parser.add_argument('-d','--device', 
                    type=str, 
                    default = 'cpu', 
                    required=False, 
                    help = 'Set device')

args = parser.parse_args()
FILE_VAR = args.name
BATCH_SIZE = args.batch_size
FORCE_COEFF = args.force_loss_coefficient
TRANSFER_LEARNING = args.transfer_learning
DEVICE = args.device
EPOCHS = args.max_epochs

device = torch.device(DEVICE)

with open('data/train_data_{}.pkl'.format(FILE_VAR), 'rb') as f:
    training, validation = pickle.load(f)

mode = None
if TRANSFER_LEARNING == 0:
    mode = 'normal'
if TRANSFER_LEARNING == 1:
    mode = 'transfer'
if TRANSFER_LEARNING == 2:
    mode = 'scratch'

model, optimlist = Model.getModel(mode, device)

        
AdamW = torch.optim.AdamW(optimlist)

AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=5, threshold=0)

latest_checkpoint = 'checkpoints/sratch-force-training_{}.pt'.format(FILE_VAR)

try:
    with open('variables/losses_{}.pkl'.format(FILE_VAR), 'rb') as f:  # Python 3: open(..., 'rb')
        losses_energy, losses_forces = pickle.load(f)
except:
    losses_energy = []
    losses_forces = []


if os.path.isfile(latest_checkpoint):
    checkpoint = torch.load(latest_checkpoint)
    try:
        model.load_state_dict(checkpoint['model'])
        AdamW.load_state_dict(checkpoint['AdamW'])
        AdamW_scheduler.load_state_dict(checkpoint['AdamW_scheduler'])
    except RuntimeError:
        pass

shft = 4.75717

def validate():

    total_mse = 0.0
    force_mse = 0.0 
    count = 0

    for properties in validation:
        species = torch.unsqueeze(torch.as_tensor(
                                properties['species']
                                ),0).to(device)
          
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).double()

        true_forces = properties['forces'].to(device).float()
        cell = properties['cell'].to(device).float()
        pbc = properties['pbc'].to(device)
        num_atoms = (species >= 0).to(device).sum(dim=1, dtype=true_energies.dtype)

        _, predicted_energies = model((species, coordinates), cell, pbc)


        forces = -torch.autograd.grad(predicted_energies.sum(), 
                                      coordinates, create_graph=True, retain_graph=True)[0]
        force_mse += (mse(true_forces, 
                      Hartree * forces ).sum(dim=(1,2)) / num_atoms ).mean()
        total_mse += mse(Hartree * predicted_energies / num_atoms, (true_energies + shft * species.shape[1])/num_atoms ).mean()
        count += predicted_energies.shape[0]

    compiled_model = torch.jit.script(model.neural_networks)
    torch.jit.save(compiled_model, 'models/test_model_{}.pt'.format(FILE_VAR))
    return math.sqrt(total_mse/count), math.sqrt(force_mse/count)

AdamW_scheduler.last_epoch = 0

mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = EPOCHS
early_stopping_learning_rate = 1.0E-7
force_coefficient = FORCE_COEFF
energy_coefficient = 1 - force_coefficient
best_model_checkpoint = 'checkpoints/scratch-force-training_best_{}.pt'.format (FILE_VAR)

AdamW.zero_grad()

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    rmse = validate()
    losses_energy.append(rmse[0])
    losses_forces.append(rmse[1])
    print('Val RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

    learning_rate = AdamW.param_groups[0]['lr']

    if AdamW_scheduler.is_better(rmse[0], AdamW_scheduler.best):
        torch.save(model.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse[0])

    cum_loss = 0
    l = []
    f = 0
    for i, properties in tqdm(enumerate(training.shuffle()),total=len(training), desc="epoch {}".format(AdamW_scheduler.last_epoch)):
        species = torch.unsqueeze(torch.as_tensor(properties['species']),0).to(device)
        coordinates = properties['coordinates'].to(device).float().             requires_grad_(True)
        true_energies = properties['energies'].to(device).double()
        true_forces = properties['forces'].to(device).float()
        cell = properties['cell'].to(device).float()
        pbc = properties['pbc'].to(device)
        num_atoms = (species >= 0).to(device).sum(dim=1, 
                                                  dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates), 
                                       cell = cell, 
                                       pbc = pbc)
        forces = -torch.autograd.grad(predicted_energies, coordinates, create_graph=True, retain_graph=True)[0]
        energy_loss = (mse(Hartree * predicted_energies / num_atoms.item(), (true_energies + shft * species.shape[1]) / num_atoms )).mean()
        force_loss = (mse(true_forces, Hartree * forces ).sum(dim=(1,2) ) 
                          / num_atoms).mean()
        loss = (energy_coefficient*energy_loss + force_coefficient * force_loss) / BATCH_SIZE
        cum_loss += energy_loss
        f += force_loss

        loss.backward()
        if i % BATCH_SIZE == 0:
            AdamW.step()
            AdamW.zero_grad()
    print('Training RMSE:', cum_loss.sqrt().item()/np.sqrt(len(training)), f.sqrt().item() / np.sqrt(len(training) ))    
    torch.save({
        'model': model.state_dict(),
        'AdamW': AdamW.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
    }, latest_checkpoint)
