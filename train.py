'''Treniramo na 50% splita'''
import torch
import torchani
import os
import math
#import torch.utils.tensorboard
import tqdm
import numpy as np
from ase.io.trajectory import Trajectory
from ase.io.trajectory import TrajectoryWriter
import torchani
import torchani.data
from tqdm import tqdm
import matplotlib.pyplot as plt
#import h5py
from scipy.stats import linregress
import pickle
import copy


from ase.units import Hartree


# Import the library
import argparse# Create the parser
parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('-n','--name', type=str, required=True, help='Model name')
parser.add_argument('-d','--data-file', type=str, required=False, help='Path to dataset in .pkl format (default is data/train_data_{name}.pkl)')
parser.add_argument('-m','--model-dir', type=str, required=False, help='Save model dir (default is models)')
parser.add_argument('-c','--checkpoint-dir', type=str, required=False, help='Checkpoint dir (default is checkpoints)')
parser.add_argument('-v','--var-dir', type=str, required=False, help='Varialbes dir (default is variables)')
parser.add_argument('-t','--transfer-learning', type=bool, required=False, help='If set to True, top and bottom layer will be frozen (default is False)')
parser.add_argument('-b','--batch-size', type=int, required=False, help = 'Set batch size (default is 1)')
parser.add_argument('-r','--force-energy-loss-ratio', type=float, default = 9.0, required=False, help = 'Force loss to energy loss ratio (default is 9.0)')# Parse the argument
args = parser.parse_args()# Print "Hello" + the user input argument

FILE_VAR = args.name
BATCH_SIZE = args.batch_size
RATIO = args.force_energy_loss_ratio

device = torch.device('cuda')

with open('data/train_data_{}.pkl'.format(FILE_VAR), 'rb') as f:
    training, validation = pickle.load(f)
    
#DEFINIRAJ MODEL
#energy_shifts = np.aAdamWrray([-1.11902447,  -2.35066639 , -4.42472954 , -3.20551635 ,  -5.05470390 , -2.17385848,  -4.72531884])
energy_shifts = np.array([0,  0 , 0 , 0 ,  0 , 0,  0,0,0,0]) 
#energy_shifts = np.array([-6.883758738459247,  -6.883758738459247 , -6.883758738459247 , -6.883758738459247 ,  -6.883758738459247 , -6.883758738459247,  -6.883758738459247])
energy_shifter = torchani.utils.EnergyShifter(energy_shifts/Hartree, True).to(device)
model = torchani.models.ANI2x().ase().model.to(device)
for _, member in enumerate(model):
    member.neural_networks['S'] = copy.deepcopy(member.neural_networks.C.to(device))
    #member.neural_networks['I'] = copy.deepcopy(member.neural_networks.Cl.to(device))
    #member.neural_networks['Cu'] = copy.deepcopy(member.neural_networks.S.to(device)) 
model.energy_shifter = energy_shifter.to(device)
model = model.to(device)

ll = []
for k in training:
    ll.append(k['energies'].item()/k['species'].shape[0] )
print(np.mean(ll), np.var(ll))


#ZAMRZNI SLOJEVE
cnt = 0
optimlist = []
layers = [0,1,2,3,4,5,6,7]
lrs = {0: 10**-4, 1: 10**-4, 2: 10**-4, 3: 10**-4, 4: 10**-4, 5: 10**-4, 6: 10**-4, 7: 10**-4}

#for k in model.neural_networks:
#    for j in k:
#        for l in k[j].parameters():
#            cnt+=1
#            l.requires_grad = True
#            optimlist.append({'params': l, 'weight_decay': 0.1, 'lr' :  lrs[cnt%8]})

for k in model.parameters():
    #if cnt % 2 != 0:
    #    k.requires_grad = False
    #elif cnt % 8 == 0 or (cnt-1) % 8 == 0:
    #    k.requires_grad = False
    if cnt % 8 not in layers:
        k.requires_grad = False
    else:
        k.requires_grad = True
        optimlist.append({'params': k, 'weight_decay': 0.1, 'lr' :  lrs[cnt%8]})
    cnt += 1
        
print('Broj slojeva', cnt)
#for k in model.parameters():
#    print(k.requires_grad)
        
AdamW = torch.optim.AdamW(optimlist)

AdamW_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(AdamW, factor=0.5, patience=5, threshold=0)

latest_checkpoint = 'checkpoints/force-training_{}.pt'.format(FILE_VAR)

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

energy_shifts = np.array([-1.11902447,  -2.35066639 , -4.42472954 , -3.20551635 ,  -5.05470390 , -2.17385848,  -4.72531884, 0, 0, 0])
energy_shifter = torchani.utils.EnergyShifter(None).to(device)
shft = 4.75717 + 0.2379

def validate():

    # run validation
    mse_sum = torch.nn.MSELoss(reduction='sum')
    total_mse = 0.0
    force_mse = 0.0 
    count = 0
    ll = []
    x = []
    y = []
    
    for properties in validation:
        species = torch.unsqueeze(torch.as_tensor(properties['species']),0).to(device)
        
        
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).double()
        #true_energies /= Hartree
        true_forces = properties['forces'].to(device).float()
        cell = properties['cell'].to(device).float()
        pbc = properties['pbc'].to(device)
        num_atoms = (species >= 0).to(device).sum(dim=1, dtype=true_energies.dtype)
        #es = energy_shifter.sae(species)
        #print(es)
        #print(cell)

        # Now the total loss has two parts, energy loss and force loss
        #print(model.aev_computer((species, coordinates), cell, pbc).aevs.shape)
        _, predicted_energies = model((species, coordinates), cell, pbc)
        #print('Pfgdsfadfasdfasdfgdasfdasfasfaf')
        #print(Hartree * predicted_energies, true_energies + 4.49786 * species.shape[1])
        #print( (true_energies-(predicted_energies*Hartree))/species.shape[1] )
        ll.append( ((true_energies-(predicted_energies*Hartree))/species.shape[1]).item() )
        #x.append(true_energies.item())
        #y.append(predicted_energies.item())
        forces = -torch.autograd.grad(predicted_energies.sum(), coordinates, create_graph=True, retain_graph=True)[0]
        #print(forces.sum(dim = (1,2)).mean())
        #print( mse(true_forces, Hartree * forces).mean())
        force_mse += mse(true_forces/num_atoms.item(), Hartree * forces/num_atoms.item() ).mean()
        total_mse += mse_sum(Hartree * predicted_energies / num_atoms.item(), (true_energies + shft * species.shape[1])/num_atoms.item() ).item()
        count += predicted_energies.shape[0]
        #print(count)
    print('val params:', np.mean(ll) + shft, np.var(ll))
    #print(linregress(y,x).intercept, linregress(y,x).slope, linregress(y,x))
    compiled_model = torch.jit.script(model.neural_networks)
    torch.jit.save(compiled_model, 'models/compiled_model_{}.pt'.format(FILE_VAR))
    return math.sqrt(total_mse/count), math.sqrt(force_mse/count)

#tensorboard = torch.utils.tensorboard.SummaryWriter()
AdamW_scheduler.last_epoch = 0

mse = torch.nn.MSELoss(reduction='none')

print("training starting from epoch", AdamW_scheduler.last_epoch + 1)
max_epochs = 2000
early_stopping_learning_rate = 1.0E-7
force_coefficient = 0.9
energy_coefficient = 1 - force_coefficient
  # controls the importance of energy loss vs force loss
best_model_checkpoint = 'checkpoints/force-training_best_{}.pt'.format(FILE_VAR)

AdamW.zero_grad()

for _ in range(AdamW_scheduler.last_epoch + 1, max_epochs):
    #if _% 5 == 0:
    rmse = validate()
    losses_energy.append(rmse[0])
    losses_forces.append(rmse[1])
    print('Val RMSE:', rmse, 'at epoch', AdamW_scheduler.last_epoch + 1)

    learning_rate = AdamW.param_groups[0]['lr']

    #if learning_rate < early_stopping_learning_rate:
    #    break

    # checkpoint
    if AdamW_scheduler.is_better(rmse[0], AdamW_scheduler.best):
        torch.save(model.state_dict(), best_model_checkpoint)

    AdamW_scheduler.step(rmse[0])
    #SGD_scheduler.step(rmse)

    #tensorboard.add_scalar('validation_rmse', rmse, AdamW_scheduler.last_epoch)
    #tensorboard.add_scalar('best_validation_rmse', AdamW_scheduler.best, AdamW_scheduler.last_epoch)
    #tensorboard.add_scalar('learning_rate', learning_rate, AdamW_scheduler.last_epoch)
    #AdamW.zero_grad()
    # Besides being stored in x, species and coordinates are also stored in y.
    # So here, for simplicity, we just ignore the x and use y for everything.
    cum_loss = 0
    l = []
    f = 0
    for i, properties in tqdm(enumerate(training.shuffle()),total=len(training), desc="epoch {}".format(AdamW_scheduler.last_epoch)):
        species = torch.unsqueeze(torch.as_tensor(properties['species']),0).to(device)
        #print(species)
        coordinates = properties['coordinates'].to(device).float().requires_grad_(True)
        true_energies = properties['energies'].to(device).double()
        #true_energies = Hartree
        true_forces = properties['forces'].to(device).float()
        #true_forces /= Hartree
        cell = properties['cell'].to(device).float()
        pbc = properties['pbc'].to(device)
        #print(coordinates.shape)
        num_atoms = (species >= 0).to(device).sum(dim=1, dtype=true_energies.dtype)
        _, predicted_energies = model((species, coordinates), cell = cell, pbc = pbc)
        l.append( ((true_energies-(predicted_energies*Hartree))/species.shape[1]).item() )
        
        

        # We can use torch.autograd.grad to compute force. Remember to
        # create graph so that the loss of the force can contribute to
        # the gradient of parameters, and also to retain graph so that
        # we can backward through it a second time when computing gradient
        # w.r.t. parameters.
        forces = -torch.autograd.grad(predicted_energies, coordinates, create_graph=True, retain_graph=True)[0]

        # Now the total loss has two parts, energy loss and force loss
        energy_loss = (mse(Hartree * predicted_energies/ num_atoms.item(), (true_energies + shft * species.shape[1]) / num_atoms.item() )).mean()
        force_loss = (mse(true_forces / num_atoms.item(), Hartree * forces / num_atoms.item())).mean()
        loss = (energy_coefficient*energy_loss + force_coefficient * force_loss)
        cum_loss += energy_loss
        f += force_loss
        #loss.type(torch.DoubleTensor)
        

        loss.backward()

        if i % 2 == 0:
            #print(i)
            AdamW.step()
            AdamW.zero_grad()

        #SGD.zero_grad()
    print('Training RMSE:', cum_loss.sqrt().item()/np.sqrt(len(training)), f.sqrt().item() / np.sqrt(len(training) ))    
    print('Train params:', np.mean(l)+shft, np.var(l))    
    #shft = -np.mean(ll)
    #print(shft)
        
        #AdamW_scheduler.step(metrics = loss)
        #SGD.step()

        # write current batch loss to TensorBoard
        #tensorboard.add_scalar('batch_loss', loss, AdamW_scheduler.last_epoch * len(training) + i)
    #AdamW_scheduler.step(metrics = loss)
    #AdamW.step()
    torch.save({
        'model': model.state_dict(),
        'AdamW': AdamW.state_dict(),
        #'SGD': SGD.state_dict(),
        'AdamW_scheduler': AdamW_scheduler.state_dict(),
        #'SGD_scheduler': SGD_scheduler.state_dict(),
    }, latest_checkpoint)


plt.plot(losses_forces)
plt.xlabel('Epohe', 
               fontweight ='bold')
plt.ylabel('Loss', 
               fontweight ='bold')
plt.savefig('figures/samo_sile_{}.png'.format(FILE_VAR))



with open('losses_{}.pkl'.format(FILE_VAR), 'wb') as f:  # Python 3: open(..., 'wb')
    pickle.dump([losses_energy, losses_forces], f)