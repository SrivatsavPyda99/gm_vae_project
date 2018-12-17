import sklearn
import numpy as np
from sklearn.decomposition import PCA
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

root = "~/Data/MNIST"
mnist_dim = 28
flattened_dim = 28 * 28
myrange = 1000

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
mnist_data = torchvision.datasets.MNIST(root, transform=transform, download=True)
mnist_loader = torch.utils.data.DataLoader(mnist_data, 
                batch_size=1, 
                sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(mnist_data)), len(mnist_data))))

def main(args):
    
    checkpoint = args.checkpoint
    gpu_is_available = False
    if args.gpu > 0:
        gpu_is_available = True
    checkpoint_path = "../checkpoints/conv_vae_{}/checkpoint_{}/vae_ckpt_{}.pkl".format(num_hidden, checkpoint, checkpoint)

    vae = conv_vae.VAE(num_hidden,latent_dim, gpu_is_available)
    
    if gpu_is_available:
        vae = vae.cuda()

    if checkpoint > 0:
        vae.load_state_dict(torch.load(checkpoint_path))
    else:
        print("Must input a checkpoint greater than 0")
        system.exit(0)


    X = np.zeros((myrange, flattened_dim))
    Z = np.zeros((myrange, latent_dim))
    labels = np.zeros(myrange)

    for i in range(myrange):
        image, label = next(iter(mnist_loader)) # get some examples (ignore labels)

        image = image[0]
        label = label[0]
        X[i,:] = image.view(-1, flattened_dim).numpy().squeeze()
        labels[i] = label

        image = Variable(image)
        if gpu_is_available:
            image = image.cuda()

        mu, logvar = vae.encoder(image)
        z = vae.reparameterize(mu, logvar)

        if(gpu_is_available):
            Z[i,:] = z.data.cpu().squeeze()
        else:
            Z[i,:] = z.data.squeeze()

    np.save("../tests/image_data.npy", X)
    np.save("../tests/z_codes.npy", Z)
    np.szve("../tests/labels.npy", labels)

    print("shape of X: {}".format(X.shape))
    print("shape of Z: {}".format(Z.shape))
    print("shape of labels: {}".format(labels.shape))


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