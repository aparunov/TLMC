import numpy as np
from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
import torchani
from tqdm import tqdm
import h5py
#New DFT data
dft = Trajectory('dft.traj', 'r')
from collections import defaultdict
from ase.units import Hartree

#Old NN potential on huge data

#Novi kalkulator
calculator = torchani.models.ANI2x().ase()

#Baseline kalkulator
#calculator_baseline = torchani.models.ANI2x().ase()

#dft energije
dft_energies=[]
#nove energije
ani2x_energies=[]
#baseine energije
baseline_energies = []


import torch
#Učitavamo težine modela
loaded_compiled_model = torch.jit.load('compiled_model_new-v2.pt')

model = loaded_compiled_model
calculator.model.neural_networks = model.to('cpu')

#Definiramo shift
shifts = np.array([-6.883758738459247,  -6.883758738459247 , -6.883758738459247 , -6.883758738459247 ,  -6.883758738459247 , -6.883758738459247,  -6.883758738459247])
energy_shifter = torchani.utils.EnergyShifter(shifts/Hartree, fit_intercept = False)
calculator.model.energy_shifter = energy_shifter
#calculator_baseline.model.energy_shifter = energy_shifter

for i in tqdm(range(len(dft))):
    #save new DFT energies to array
    dft_energies.append(dft[i].get_potential_energy())
    dft_energies[i]/=len(dft[i].get_atomic_numbers())

    #calculate energies with NN potential for the structures used with DFT
    struktura = dft[i]
    struktura.set_calculator(calculator)
    ani2x_energies.append(struktura.get_potential_energy())
    ani2x_energies[i]/=len(struktura.get_atomic_numbers())
    #struktura.set_calculator(calculator_baseline)
    #baseline_energies.append(struktura.get_potential_energy())
    #baseline_energies[i]/=len(struktura.get_atomic_numbers())

from matplotlib import pyplot as plt

#compare predictions of NN potential to DFT data
from scipy import stats

#Linearna regresija za novi model
res = stats.linregress(dft_energies[:], ani2x_energies[:])

#Linearna regresija za baseline 
#res1 = stats.linregress(dft_energies[:], baseline_energies[:])

x = np.linspace(min(dft_energies), max(dft_energies))
y = res.slope * x + res.intercept
plt.figure(figsize=(12,12))

#Plot novi model
plt.scatter(dft_energies[:], ani2x_energies[:], c = 'red', s = 12, alpha=0.5)

#Plot baseline 
#plt.scatter(dft_energies[:], np.asarray(baseline_energies[:]), c = 'brown')
#Plot novi fit
plt.plot(x,y, label = 'fit_new', color = 'steelblue')
#x = np.linspace(min(dft_energies), max(dft_energies))
#y = res1.slope * x + res1.intercept

#Plot baseline fit
#plt.plot(x,y, label = 'fit_basesline', color = 'steelblue')
print(res)

plt.grid()
#plt.legend()
#plt.plot([-0.0005,0.0005],[-0.005,0.005])
plt.xlim(-6.89, -6.88)
plt.ylim(-6.89, -6.88)

plt.savefig('energije_novo.jpeg')
plt.show()


#not accurate enough, transfer learning with new data

#SILE - plot novih, usporedba sa starima samo za r_value linerane regresije


#loaded_compiled_model = torch.jit.load('compiled_model.pt')
#model = loaded_compiled_model
#calculator.model.neural_networks = model
#shifts = np.array([-2.35066639, -2.17385848, -4.42472954, -3.20551635, -1.11902447,  -5.05470390]) 
#energy_shifter = torchani.utils.EnergyShifter(shifts/Hartree, fit_intercept=False)
#calculator.model.energy_shifter = energy_shifter
#calculator_baseline.model.energy_shifter = energy_shifter
#energy_shifter = torchani.utils.EnergyShifter(torch.Tensor([-0.5978583943827134, -38.08933878049795, -54.711968298621066, -75.19106774742086,
#-398.1577125334925, -99.80348506781634]))
#calculator.model.energy_shifter = energy_shifter


dft_forces = []
ani2x_forces=[]
#baseline_forces=[]



for i in tqdm(range(len(dft))):
    #save new DFT energies to array

    #dftF = np.asarray(dft[i].get_forces()).reshape(3,-1)
    dft_forces.append(dft[i].get_forces())
    
    #print(dft_energies[i])
    #for atoms, shifts in asft:
    #    dft_energies[i] -= np.count_nonzero(dft[i].get_atomic_numbers()==atoms)*shifts/27.2144
    #dft_energies[i]/=len(dft[i].get_atomic_numbers())
    #print(dft_energies[i])

    #calculate energies with NN potential for the structures used with DFT
    struktura = dft[i]
    struktura.set_calculator(calculator)
    ani2x_forces.append(struktura.get_forces())

    #struktura.set_calculator(calculator_baseline)
    #baseline_forces.append(struktura.get_forces())

    #struktura.set_calculator(calculator_baseline)
    #dftF = np.asarray(struktura.get_forces()).reshape(3,-1)
    #np.concatenate(baseline_forces[0], dftF[0])
    #np.concatenate(baseline_forces[1], dftF[1])
    #np.concatenate(baseline_forces[2], dftF[2])
    #for atoms, shifts in asft:
    #    ani2x_energies[i] -= np.count_nonzero(dft[i].get_atomic_numbers()==atoms)*shifts/27.2144
    #ani2x_energies[i]/=len(dft[i].get_atomic_numbers())

#compare predictions of NN potential to DFT data
print(len(dft_forces))
rv = 0
#rv_base  = 0
for i in range(len(dft)):
    #print(np.asarray(dft_forces[i]).shape, np.asarray(ani2x_forces[i]).shape  )
    #print(len(dft_forces[i][0]))
    rv += stats.linregress(dft_forces[i][0], ani2x_forces[i][0]).rvalue
    #rv_base += stats.linregress(dft_forces[i][0], baseline_forces[i][0]).rvalue
    plt.scatter(dft_forces[i], ani2x_forces[i], s = 5, alpha = 0.3)
    #plt.scatter(dft_forces[i], baseline_forces[i])
    #plt.plot([-0.0005,0.0005],[-0.005,0.005])
    #plt.xlim(-2100, -2000)
    #plt.ylim(-2200, -2000)   
    #plt.savefig('unzoom_sile_' + str(i) + '.jpeg')
print(rv/len(dft))
#plt.xlim(-0.01, 0.01)
#plt.ylim(-0.01, 0.01)      
plt.grid()
plt.savefig('sile_novo.png')


