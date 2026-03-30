import numpy as np 
import pandas as pd 
import geopandas as gpd
import utils.cross_validation as cval
import torch
from torch import nn
import matplotlib as plt
from captum.attr import IntegratedGradients


#Neural network
class MLP(nn.Module):
   
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential( #Passes through layers sequentially
             nn.Linear(17, 30, bias= True), #100 inputs into 200 perceptrons
             nn.ReLU(), #Activation function
             nn.Linear(30, 100, bias= True), # 200 inputs 100 outputs
             nn.ReLU(),
             nn.Linear(100, 2, bias= True), # 200 inputs 100 outputs
             #nn.Dropout(0.05)
         )

    def forward(self, x): #Feed input X into layers mentioned above
      return self.layers(x)
    
    
if __name__ == "__main__":

    model = MLP()

    model.eval()
    input = torch.rand(1195, 1122)
    baseline = torch.zeros(1195, 1122)

    ig = IntegratedGradients(model)
    attributions, delta = ig.attribute(input, baseline, target=0, return_convergence_delta=True)
    attr = attributions.detach().numpy()