import torch
import argparse# Create the parser


parser = argparse.ArgumentParser()# Add an argument
parser.add_argument('-n','--name', type=str, required=True, help='Model name')
args = parser.parse_args()# Print "Hello" + the user input argument
FILE_VAR = args.name

loaded_compiled_model = torch.jit.load('../models/compiled_model_{}.pt'.format(FILE_VAR))
model = loaded_compiled_model.to('cpu')
torch.jit.save(model,'../models/cpu_{}.pt'.format(FILE_VAR))