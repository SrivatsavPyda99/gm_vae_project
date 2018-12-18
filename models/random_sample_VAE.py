import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
from torch.autograd import Variable
from conv_vae import VAE

vae = VAE(64, 100, True)
vae.load_state_dict(torch.load("../checkpoints/conv_vae_64/checkpoint_800000/vae_ckpt_800000.pkl"))

Z = np.load("../tests/z_codes.npy")
Z = torch.Tensor(Z[:,:,np.newaxis,np.newaxis]).cuda()

X = np.zeros((1024, 784))

for i in range(8):
    inp = Variable(Z[i * 128: (i +1) * 128,:,:,:])
    out = vae.decoder(inp)
    out = out.view(-1, 784)
    X[i * 128: (i +1) * 128,:] = out.data.cpu().numpy().squeeze()

np.save("../tests/image_data_VAE.npy", X)