import numpy as np
from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
import torchani
from tqdm import tqdm
import h5py
import pandas as pd
#New DFT data
from collections import defaultdict
from ase.units import Hartree
import seaborn as sns
import torch

import pandas as pd

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-n','--name', type=str, required=True, help='Model name')
args = parser.parse_args()

FILE_VAR = args.name

dft = Trajectory('data/trajs/{}.traj'.format(FILE_VAR), 'r')

lst = defaultdict()
dct = defaultdict()
dct2 = defaultdict()
dct3 = defaultdict()
dct4 = defaultdict()
cnt = 0
num_atoms = []
for id, k in enumerate(dft):
    name = k.get_chemical_formula()
    atoms = np.unique(k.get_chemical_symbols())
    if k.get_global_number_of_atoms() not in num_atoms:
        num_atoms.append(k.get_global_number_of_atoms())
    dct3[id] = k.get_potential_energy()/k.get_global_number_of_atoms()
    forces = torch.sqrt(torch.sum( 
        torch.pow(torch.as_tensor(k.get_forces()), 2),1 ) )
    for idx, _ in enumerate(forces):
        #print(idx)
        dct4[cnt] = id
        dct2[cnt] = k.get_chemical_symbols()[idx]
        dct[cnt] = forces[idx].item()
        cnt += 1
    #print(dct['energies'])
    lst['col' + str(id)] = dct
#print(len(dct) )
DF1= pd.DataFrame.from_dict(dct, orient = 'index')
DF2= pd.DataFrame.from_dict(dct2, orient = 'index')  
DF3= pd.DataFrame.from_dict(dct4, orient = 'index')  
DF = pd.DataFrame.from_dict(dct3, orient= 'index')
DF.columns = ['Energies [eV]']

df = pd.concat([DF1,DF2, DF3], axis = 1) 
df.columns = [ 'Forces [eV/$\AA$]', 'atoms', 'structure_id']
mn = df['Forces [eV/$\AA$]'].mean()
std = df['Forces [eV/$\AA$]'].std()
print(df.describe())
print( 'Outlier structures:', len(df.loc[df['Forces [eV/$\AA$]'] > mn+5*std]['structure_id'].unique()), 'out of:', len(dft)  )
print( 'Outlier atoms:', len(df.loc[df['Forces [eV/$\AA$]'] > mn+5*std]) )

font = {
    'savefig.bbox': 'tight',
    }


sns.set(font_scale =  1.2, style = 'white', palette = 'viridis', rc = font)
sns.displot(DF, x="Energies [eV]", fill = 'True', binwidth = 0.002).savefig("figures/energies_{}.png".format(FILE_VAR))
dspl = sns.displot(df, x="Forces [eV/$\AA$]", kind = 'hist', multiple = 'stack', binwidth = 0.1, hue = 'atoms').set(xlim = (0,2))
dspl.savefig("figures/forces_{}.png".format(FILE_VAR))