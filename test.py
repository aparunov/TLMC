import numpy as np
from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
import torchani
from tqdm import tqdm
from collections import defaultdict
from ase.units import Hartree
import math
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl


dft_energies=[]
ani2x_energies=[]
baseline_energies = []


import torch
import pickle

import argparse# Create the parser
parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('-n','--name', type=str, required=True, help='Model name')
args = parser.parse_args()# Print "Hello" + the user input argument

FILE_VAR = args.name

with open('data/train_data_{}.pkl'.format(FILE_VAR), 'rb') as f:
    training, validation = pickle.load(f)

device = torch.device('cpu')

#Učitavamo težine modela
loaded_compiled_model = torch.jit.load('models/test_model_{}.pt'.format(FILE_VAR)).to(device)
model = torchani.models.ANI2x().ase().model.to(device)
baseline_model = torchani.models.ANI2x().ase().model.to(device)
nn = loaded_compiled_model.to(device)

model.neural_networks = nn
energy_shifts = np.array([0,  0 , 0 , 0 ,  0 , 0,  0,0,0,0]) 
energy_shifter = torchani.utils.EnergyShifter(energy_shifts/Hartree, True).to(device)
model.energy_shifter = energy_shifter.to(device)
baseline_model.energy_shifter = energy_shifter.to(device)

#Definiramo shift
shft = 4.75717  #+ 0.2379

mse = torch.nn.MSELoss(reduction='none')
mse_sum = torch.nn.MSELoss(reduction='sum')
total_mse = 0.0
force_mse = 0.0
total_mse_base = 0.0
force_mse_base = 0.0  
count = 0
cnt = 0
dct = defaultdict()
dct2 = defaultdict()
dct3 = defaultdict()
dct4 = defaultdict()
dct5 = defaultdict()
dct6 = defaultdict()

for idx, properties in enumerate(tqdm(validation) ):
        species = torch.unsqueeze(torch.as_tensor(properties['species']),0).to(device)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).double()
        
        #true_energies /= Hartree
        true_forces = properties['forces'].to(device).float()
    
        cell = properties['cell'].to(device).float()
        pbc = properties['pbc'].to(device)
        #print(species)
        num_atoms = (species >= 0).to(device).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates), cell, pbc)
        _, baseline_energies = baseline_model((species, coordinates), cell, pbc)
        dct3[idx] = (predicted_energies.item() )/num_atoms.item()*Hartree
        dct5[idx] = (baseline_energies.item() )/num_atoms.item()*Hartree
        dct4[idx] = (true_energies.item()  + shft*num_atoms.item() )/num_atoms.item()
        baseline_forces = -torch.autograd.grad(baseline_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
        
        forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
        abs_forces_ani = torch.sqrt(torch.sum(torch.pow(baseline_forces, 2), (0,2) ))
        abs_forces_true = torch.sqrt(torch.sum(torch.pow(true_forces, 2), (0,2) ))
        for jdx, k in enumerate(torch.sqrt(torch.sum(torch.pow(forces, 2), (0,2) ))):
            #print(k.item()/num_atoms)
            dct2[cnt] = k.item()*Hartree
            dct6[cnt] = abs_forces_ani[jdx].item()
            dct[cnt] = abs_forces_true[jdx].item()
            cnt += 1
        #print(predicted_energies, true_energies + shft * species.shape[1] )
        force_mse_base += (mse(true_forces, Hartree * baseline_forces ).sum(dim=(1,2) ) / num_atoms).mean()
        
        force_mse += (mse(true_forces, Hartree * forces ).sum(dim=(1,2) ) / num_atoms).mean()
        total_mse_base += mse_sum(Hartree * baseline_energies / num_atoms.item(), (true_energies + shft * species.shape[1])/num_atoms.item() ).item()
        total_mse += mse_sum(Hartree * predicted_energies / num_atoms.item(), (true_energies + shft * species.shape[1])/num_atoms.item() ).item()
        count += predicted_energies.shape[0]
print('RMSE ENERGY (BASELINE | FINE-TUNED):', math.sqrt(total_mse_base/count) , '|',  math.sqrt(total_mse/count))
print('RMSE FORCE (BASELINE | FINE-TUNED):', math.sqrt(force_mse_base/count) , '|'  , math.sqrt(force_mse/count))



forces_base = pd.DataFrame.from_dict(dct6, orient='index')
forces_pred = pd.DataFrame.from_dict(dct, orient='index')
forces_true = pd.DataFrame.from_dict(dct2, orient='index')
forces_df = pd.concat([forces_pred, forces_true, forces_base], 1)



engs_base = pd.DataFrame.from_dict(dct5, orient='index') 
engs_pred = pd.DataFrame.from_dict(dct3, orient='index') 
engs_true = pd.DataFrame.from_dict(dct4, orient='index') 
engs_df = pd.concat([engs_pred, engs_true, engs_base], 1)
#engs_df.sub(engs_df.mean(axis=1), axis=0)
engs_df.columns = ['Fine tuned [eV]', 'DFT [eV]', 'ANI2x [eV]']
forces_df.columns = ['Fine tuned [eV/$\AA$]', 'DFT [eV/$\AA$]', 'ANI2x [eV/$\AA$]']
print('Energies correlation matrtix:')
print(engs_df.corr())
pd.set_option("display.max_rows", None, "display.max_columns", None)
print('Forces correlation matrtix:')
print(forces_df.corr())

print(forces_df.describe())

#print('PARAMETRI', mpl.rcParams.keys())
font = {
    'axes.titlesize' : 40,
    'axes.labelsize' : 35,
    'lines.linewidth' : 10, 
    'lines.markersize' : 15,
    'savefig.bbox': 'tight',
    }

with mpl.rc_context(font):
    plt.rcParams.update(font)
    sns.set(font_scale =  3.05, style = 'ticks', palette = 'dark', rc = font)
    fig, axs = plt.subplots(ncols=2, nrows = 2, figsize = (30,30))
    #scatter_kws={"color": "navy"}, line_kws={"lw" : 6, "color": "darkgray"}
    sns.regplot(data = forces_df, x="Fine tuned [eV/$\AA$]", y="DFT [eV/$\AA$]", ci = None, scatter_kws={"color": "navy"}, line_kws={"lw" : 7, "color": "gray"},  ax = axs[0][1])
    sns.regplot(data = forces_df, x="ANI2x [eV/$\AA$]", y="DFT [eV/$\AA$]", ci = None,scatter_kws={"color": "navy"}, line_kws={"lw" : 7, "color": "gray"},  ax = axs[0][0])
    sns.regplot(data = engs_df, x="Fine tuned [eV]", y="DFT [eV]", ci = None,scatter_kws={"color": "navy"}, line_kws={"lw" : 7, "color": "gray"},  ax = axs[1][1])
    sns.regplot(data = engs_df, x="ANI2x [eV]", y="DFT [eV]", ci = None,scatter_kws={"color": "navy"}, line_kws={"lw" : 7, "color": "gray"}, ax = axs[1][0])
    axs[0][0].title.set_text('a)')
    axs[0][1].title.set_text('b)')
    axs[1][0].title.set_text('c)')
    axs[1][1].title.set_text('d)')
    fig.savefig("figures/test_forces_{}.png".format(FILE_VAR))

