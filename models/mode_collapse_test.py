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
import conv_vae



num_hidden= 64
latent_dim = 100
root = "~/Data/MNIST"
training_size = 60000

def main(args):
    
    checkpoint = args.checkpoint
    gpu_is_available = False
    if args.gpu > 0:
        gpu_is_available = True
    checkpoint_path = "../checkpoints/conv_vae_{}/checkpoint_{}/vae_ckpt_{}.pkl".format(num_hidden, checkpoint, checkpoint)
        
        
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    mnist_data = torchvision.datasets.MNIST(root, transform=transform, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_data, 
                    batch_size=1, 
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(mnist_data)), 
                                                                                        training_size)))
    vae = conv_vae.VAE(num_hidden,latent_dim, gpu_is_available)

    
    if gpu_is_available:
        vae = vae.cuda()

    if checkpoint > 0:
        vae.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Must input a checkpoint greater than 0")
        system.exit(0)

    total_measures = torch.zeros((10, latent_dim))
    z_measures = []
    for i in range(10):
        list_i = []
        z_measures.append(list_i)

    for i in range(len(mnist_data)):
        image, label = next(iter(mnist_loader)) # get some examples (ignore labels)
        image = Variable(images)
        mu, logvar = vae.encoder(image)
        z = vae.reparameterize(mu, logvar)

        if(gpu_is_available):
            total_measures[label[0],:] += z.data.cpu().squeeze().numpy()
            list_i[label[0]].append(z.data.cpu().squeeze().numpy())
        else:
            total_measures[label[0],:] += z.data.squeeze().numpy()
            list_i[label[0]].append(z.data.squeeze().numpy())

    np.save(os.path.join(base_dir, "total_measures.npy"), total_measures)
    np.save(os.path.join(base_dir, "z_measures.npy"), z_measures)
    '''
    avg_measures = total_measures/len(mnist_data)
    for i in range(10):
        print("average z code for a {}: {}".format(i, avg_measures[i,:]))

    sigma_measures
    
    for i in range(10):
        for j in range(len(z_measures[i]))
        
    
    for i in range(10):
        print("some ")
    '''
        

    




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=int, default=0)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    args = parser.parse_args()
    main(args)
