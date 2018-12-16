from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


import utils

flat_img_size = 28 * 28
mnist_dim = 28
mm=0.8
slope_leaky_relu = 0.2

class encoder(nn.Module):
    def __init__(self, layer_multiplier=64, latent_dim=100):
        super(encoder, self).__init__()

        self.latent_dim = latent_dim
        self.conv1 = nn.Conv2d(1, layer_multiplier, 13, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_multiplier, momentum=mm)
        self.conv2 = nn.Conv2d(layer_multiplier, layer_multiplier * 2, 9, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(layer_multiplier*2, momentum=mm)
        self.conv3 = nn.Conv2d(layer_multiplier*2, layer_multiplier * 4, 5, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(layer_multiplier*4, momentum=mm)
        self.conv4 = nn.Conv2d(layer_multiplier*4, layer_multiplier * 8, 3, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(layer_multiplier*8, momentum=mm)
        self.conv51 = nn.Conv2d(layer_multiplier*8, latent_dim, 2, 1, 0, bias=False)
        self.conv52 = nn.Conv2d(layer_multiplier*8, latent_dim, 2, 1, 0, bias=False)

    def forward(self, x):
        """
        Assumes x is an image
        """
        x = self.bn1(F.leaky_relu(self.conv1(x), slope_leaky_relu))
        x = self.bn2(F.leaky_relu(self.conv2(x), slope_leaky_relu))
        x = self.bn3(F.leaky_relu(self.conv3(x), slope_leaky_relu))
        x = self.bn4(F.leaky_relu(self.conv4(x), slope_leaky_relu))
        mu = self.conv51(x)
        logvar = self.conv51(x)
        return mu, logvar

class decoder(nn.Module):
    def __init__(self, layer_multiplier=64, latent_dim=100):
        super(decoder, self).__init__()
        self.latent_dim = latent_dim

        self.conv1 = nn.ConvTranspose2d(latent_dim, layer_multiplier * 8, 2, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_multiplier * 8, momentum=0.8)
        self.conv2 = nn.ConvTranspose2d(layer_multiplier * 8, layer_multiplier * 4, 3, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(layer_multiplier * 4, momentum=0.8)
        self.conv3 = nn.ConvTranspose2d(layer_multiplier * 4, layer_multiplier * 2, 5, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(layer_multiplier * 2, momentum=0.8)
        self.conv4 = nn.ConvTranspose2d(layer_multiplier * 2, layer_multiplier, 9, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(layer_multiplier, momentum=0.8)
        self.conv5 = nn.ConvTranspose2d(layer_multiplier, 1, 13, 1, 0, bias=False)

    def forward(self, x):
        x = self.bn1(F.relu(self.conv1(x)))
        x = self.bn2(F.relu(self.conv2(x)))
        x = self.bn3(F.relu(self.conv3(x)))
        x = self.bn4(F.relu(self.conv4(x)))
        x = self.conv5(x)
        return x

class VAE(nn.Module):

  def __init__(self, num_hidden, latent_dim, gpu_is_available):
    super(VAE, self).__init__()

    self.latent_dim = latent_dim
    self.gpu_is_available = gpu_is_available
    self.decoder = decoder(layer_multiplier=num_hidden, latent_dim=latent_dim)
    self.encoder = encoder(layer_multiplier=num_hidden, latent_dim=latent_dim)

  def sample(self, mu, logvar):
    eps = Variable(torch.randn(mu.size()))
    if self.gpu_is_available:
        eps = eps.cuda()

    z = eps.mul(logvar.mul(0.5).exp_()).add_(mu)
    logqz = utils.log_normal(z, mu, logvar)

    zeros = Variable(torch.zeros(z.size()))
    if self.gpu_is_available:
        zeros = zeros.cuda()

    logpz = utils.log_normal(z, zeros, zeros)

    return z, logpz, logqz

  def forward(self, x, k=1, warmup_const=1.):
    #x = x.repeat(k, 1)
    mu, logvar = self.encoder(x)
    z, logpz, logqz = self.sample(mu, logvar)
    x_logits = self.decoder(z)

    logpx = utils.log_bernoulli(x_logits.view(-1, flat_img_size), 
                                x.squeeze().view(-1, flat_img_size))

    logpx = torch.mean(logpx)
    logpz = torch.mean(logpz)
    logqz = torch.mean(logqz)


    elbo = logpx + logpz - warmup_const * logqz

    # need correction for Tensor.repeat
    #elbo = utils.log_mean_exp(elbo.view(k, -1).transpose(0, 1))
    #elbo = torch.mean(elbo)

    #logpx = torch.mean(logpx)
    #logpz = torch.mean(logpz)
    #logqz = torch.mean(logqz)

    return elbo, logpx, logpz, logqz, x_logits