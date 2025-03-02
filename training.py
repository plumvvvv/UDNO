from phase1training import trainingloop as trainingloop1
from phase2training import trainingloop as trainingloop2
from phase1net import SpectralNet
import torch
import os

datapath = "/home/ml4a/code/dl-spectral-graph-partitioning"
modelpath = "/home/ml4a/code/dl-spectral-graph-partitioning/reproducedmodels"
device =  'cpu'#'GPU' if torch.cuda.is_available() else 'CPU'

spectralnet_path = os.path.join(modelpath,"spectralnet.pth")
trainingloop1(datapath,spectralnet_path,device)
f = SpectralNet().to(device)
f.load_state_dict(torch.load(spectralnet_path))
f.eval()
for p in f.parameters():
    p.requires_grad = False
f.eval()

partitionnet_path = os.path.join(modelpath,"partitionnet.pth")
trainingloop2(f,datapath,partitionnet_path,device)
