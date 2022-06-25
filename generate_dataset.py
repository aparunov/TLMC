'''Treniramo na 50% splita'''
import torch
import torchani

import numpy as np
from ase.io.trajectory import Trajectory
import pickle

import argparse# Create the parser
parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('-n','--name', type=str, required=True, help='Model name')
parser.add_argument('-ts', '--train-test-split', 
type=float, required=True, help = 'Percentage of data used for training')
args = parser.parse_args()

FILE_VAR = args.name
TTS = args.train_test_split

device = torch.device('cuda')


dft = Trajectory('data/trajs/{}.traj'.format(FILE_VAR), 'r')
from collections import defaultdict
lst = []
st = dict()
shft = -6.883758738459247

for k in dft:
    name = k.get_chemical_formula()
    dct = defaultdict()
    dct['species'] = k.get_chemical_symbols()
    for i, _ in enumerate(dct['species']):
        if dct['species'][i] == 'Si':
            dct['species'][i] = 'S'
        if dct['species'][i] == 'I':
            dct['species'][i] = 'Cl'
        if dct['species'][i] == 'Cu':
            dct['species'][i] = 'S'
    #print(dct['species'])
    #exit(0)

    dct['coordinates'] = torch.unsqueeze(torch.as_tensor(k.get_positions() ), 0)
    dct['energies'] = torch.unsqueeze(torch.as_tensor(k.get_potential_energy() ), 0) #- shft*len(k.get_chemical_symbols())
    dct['forces'] = torch.unsqueeze(torch.as_tensor(k.get_forces() ), 0)
    dct['cell'] = torch.as_tensor(k.get_cell())
    dct['pbc'] =  torch.as_tensor(k.get_pbc())
    #print(dct['energies'])

    lst.append(dct)

energy_shifts = np.array([-1.11902447,  -2.35066639 , -4.42472954 , -3.20551635 ,  -5.05470390 , -2.17385848,  -4.72531884])
#energy_shifts = np.array([0,  0 , 0 , 0 ,  0 , 0,  0])
#energy_shifts/=Hartree
print(energy_shifts)

tri = torchani.data.TransformableIterable(lst, ('subtract_self_energies', 'species_to_indices', 'shuffle', 'split','collate','cache') )

training, validation = tri.subtract_self_energies({'H':energy_shifts[0], 'C':energy_shifts[1], 'N':energy_shifts[2], 'O':energy_shifts[3],
'S':energy_shifts[4], 'F':energy_shifts[5], 'Cl':energy_shifts[6]}  ).species_to_indices(['H', 'C','N', 'O', 'S', 'F', 'Cl']).shuffle().split(TTS, None)
#energy_shifts-=3.92/Hartree
print(len(training), len(validation))
print(next(iter(training))['energies'])

with open('data/train_data_{}.pkl'.format(FILE_VAR), 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([training, validation], f)