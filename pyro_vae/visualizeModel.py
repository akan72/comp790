import torch
import matplotlib.pyplot as plt


modelDict = torch.load('vae_mnist003.pt')
print(modelDict.keys())

stateDict = torch.nn.Module.load_state_dict(modelDict)