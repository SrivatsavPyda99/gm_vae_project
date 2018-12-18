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
from dcgan_gpu import Generator

G = Generator(64).cuda()
G = torch.load_state_dict(torch.load("../../ul-gans-project/Pyda/checkpoints/dc_gen_64_disc_64/checkpoint_30000/generator_ckpt_30000.pkl"))

X = np.zeros((1024, 784))
def sample_z(m=128, k=100):
    """
    m = number of samples (batch size)
    k = dimension per sample (should probably be around 100)
    returns a numpy array of size m*k of (gaussian) noise to be input to the generator
    """
    return Variable(torch.randn(m, k, 1, 1).cuda())

for i in range(8):
    inp = sample_z()
    out = G(inp)
    out = out.view(-1, 784)
    X[i * 128: (i +1) * 128,:] = out.data.cpu().numpy().squeeze()

np.save("../tests/image_data_GAN.npy", X)