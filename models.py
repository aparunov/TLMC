from typing import List, Tuple

import torch 
import torchani 
import numpy as np

class Model:
    def getModel(mode: str, device: str) -> Tuple:
        '''
        Helper function that returns model and corresponding list of 
        parameters that should be optimized. 
        mode: 3 modes are available: 'normal',
        'transfer', and 'scratch'. If set to 'transer' first and final layers 
        are frozen. If set to 'scratch' weights are initialized from normal 
        distribution. 
        device: 'cpu' or 'cuda'.  
        '''
        energy_shifts = np.array([0,  0 , 0 , 0 ,  0 , 0,  0]) 
        energy_shifter = torchani.utils.EnergyShifter(
                                                    energy_shifts, 
                                                    True
                                                    ).to(device)
        model = torchani.models.ANI2x().ase().model.to(device)
        model.energy_shifter = energy_shifter.to(device)
        normal_layers = [0,1,2,3,4,5,6,7]
        transfer_layers = [2,3,4,5]
        optimlist = []
        lrs = {
            0: 10**-4, 1: 10**-4, 2: 10**-4, 
            3: 10**-4, 4: 10**-4, 5: 10**-4, 
            6: 10**-4, 7: 10**-4 
            }
        if mode == 'normal':
            layers = normal_layers
            cnt = 0
            for k in model.parameters():
                if cnt % 8 not in layers:
                    k.requires_grad = False
                else:
                    k.requires_grad = True
                    optimlist.append({'params': k, 
                                    'weight_decay': 0.1, 
                                    'lr' :  lrs[cnt%8]})
                cnt += 1
        elif mode == 'transfer':
            layers = transfer_layers
            cnt = 0
            for k in model.parameters():
                if cnt % 8 not in layers:
                    k.requires_grad = False
                else:
                    k.requires_grad = True
                    optimlist.append({'params': k, 
                                    'weight_decay': 0.1, 
                                    'lr' :  lrs[cnt%8]})
                cnt += 1
        elif mode == 'scratch':
            layers = normal_layers
            cnt = 0
            for k in model.parameters():
                torch.nn.init.normal_(k, mean=0.0, std=0.1)
                optimlist.append({'params': k, 
                                'weight_decay': 0.1, 
                                'lr' :  lrs[cnt%8]})
                cnt += 1
        return (model, optimlist)
    
    