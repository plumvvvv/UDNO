#from phase1training import trainingloop as trainingloop1
from phase2training import trainingloopForPermNetwithRankloss as trainingloop2
from phase2loss import loss_softrank
from phase1net import SpectralNet
import torch
import os
import numpy as np
import random

datapath = "/home/ml4a/code/dl-spectral-graph-partitioning"
modelpath = "/home/ml4a/code/dl-spectral-graph-partitioning/softrankmodels"
device =  'cpu'#'GPU' if torch.cuda.is_available() else 'CPU'

torch.manual_seed(176364)
np.random.seed(453658)
random.seed(41884)

spectralnet_path = os.path.join(modelpath,"spectralnet.pth")
#trainingloop1(datapath,spectralnet_path,device)
f = SpectralNet().to(device)
f.load_state_dict(torch.load(spectralnet_path))
f.eval()
for p in f.parameters():
    p.requires_grad = False
f.eval()

sepnet_path = os.path.join(modelpath,"permnetlr0.0001s0.0001b1d0.001.pth")

trainingloop2(f,loss_softrank,datapath,sepnet_path,device)