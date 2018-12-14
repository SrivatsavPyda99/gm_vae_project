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

mnist_dim = 28
flat_img_size = mnist_dim*mnist_dim
k = 100 # size of input to generator
batch_size = 32
max_iterations = 1000000
lr = 0.0002
betas = (0.5, 0.999)
print_interval = 100
save_interval = 1000
probability_interval=100
num_hidden= 256
root = "~/Data/MNIST"
save_dir = "../checkpoints/vae_{}".format(num_hidden)
num_gen = 10 # number of samples to generate at save intervals
gen_steps = 1  # number of generator updates per discriminator update
training_size=60000

class decoder(nn.Module):
    def __init__(self, num_hidden):
        super(decoder, self).__init__()
        self.mnist_dim = mnist_dim
        self.fc1 = nn.Linear(k, num_hidden)
        self.bn1 = nn.BatchNorm1d(num_hidden, momentum=0.8)
        self.fc2 = nn.Linear(num_hidden, 2*num_hidden)
        self.bn2 = nn.BatchNorm1d(2*num_hidden, momentum=0.8)
        self.fc3 = nn.Linear(2*num_hidden, 4*num_hidden)
        self.bn3 = nn.BatchNorm1d(4*num_hidden, momentum=0.8)
        self.fc4 = nn.Linear(4*num_hidden, flat_img_size)

    def forward(self, x):
        x = self.bn1(F.leaky_relu(self.fc1(x), 0.2))
        x = self.bn2(F.leaky_relu(self.fc2(x), 0.2))
        x = self.bn3(F.leaky_relu(self.fc3(x), 0.2))
        x = F.sigmoid(self.fc4(x))
        x = x.view(-1, self.mnist_dim, self.mnist_dim)
        return x

class encoder(nn.Module):
    def __init__(self, num_hidden):
        super(encoder, self).__init__()
        self.fc1 = nn.Linear(flat_img_size, 2*num_hidden)
        self.fc2 = nn.Linear(2*num_hidden, num_hidden)
        self.fc31 = nn.Linear(num_hidden, k)
        self.fc32 = nn.Linear(num_hidden, k)

    def forward(self, x):
        """
        Assumes x is an image
        """
        x = x.view(-1, flat_img_size)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        logvar = F.sigmoid(self.fc31(x))
        mu = F.sigmoid(self.fc32(x))
        return mu, logvar
    
class vanilla_vae(nn.Module):
    def __init__(self, num_hidden, gpu_is_available):
        self.gpu_is_available = gpu_is_available
        super(vanilla_vae, self).__init__()
        self.encoder = encoder(num_hidden)
        self.decoder = decoder(num_hidden)
    def reparameterize(self, mu, logvar):
        # to recreate: N(0, 1) * var(e ^{0.5 * unif(0, 1)}) + unif(0, 1)
        std = torch.exp(0.5*logvar)
        eps = Variable(torch.randn(std.shape))
        if self.gpu_is_available:
            eps=eps.cuda()
        return eps.mul(std).add_(mu)
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        z = self.decoder(z)
        return z, mu, logvar

def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x.view(-1, flat_img_size), x.view(-1, flat_img_size), reduction='sum')
    BCE = F.binary_cross_entropy(recon_x.view(-1, flat_img_size), x.view(-1, flat_img_size))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    #KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE

def sample_z(m, k):
    """
    m = number of samples (batch size)
    k = dimension per sample (should probably be around 100)
    returns a numpy array of size m*k of (gaussian) noise to be input to the generator
    """
    out = Variable(torch.Tensor(np.random.normal(size=(m, k))))
    if gpu_is_available:
        out = out.cuda()
    return out

def save_checkpoint(gpu_is_available, images_remade, vae, num_gen, base_dir, save_iter, loss_log, mu_log, logvar_log):
    """
    1. Creates a new directory corresponding to the current iteration
    2. saves discriminator and generator parameters
    3. samples num_samples images from the generator
    4. saves "objective" function over time
    5. saves gradient norms over time
    6. saves discriminator accuracies on both true and fake data over time
    """
    
    base_dir = os.path.join(base_dir, "checkpoint_{}".format(save_iter))
    vae_ckpt = os.path.join(base_dir, "vae_ckpt_{}.pkl".format(save_iter))
    reconstructed_images_dir = os.path.join(base_dir, "reconstructed_images")
    new_images_dir = os.path.join(base_dir, "new_images")
    
    
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    if not os.path.exists(reconstructed_images_dir):
        os.mkdir(reconstructed_images_dir)
    if not os.path.exists(new_images_dir):
        os.mkdir(new_images_dir)
        
    # Stuff to save in base directory
    
    torch.save(vae.state_dict(), vae_ckpt)
    np.save(os.path.join(base_dir, "loss_log.npy"), loss_log)
    np.save(os.path.join(base_dir, "mu_log.npy"), mu_log)
    np.save(os.path.join(base_dir, "logvar_log.npy"), logvar_log)
    
    # Reconstructed images
    indices = np.random.choice(range(images_remade.shape[0]), num_gen)
    for j in range(num_gen):
        index = indices[j]
        if gpu_is_available:
            img = images_remade[index].data.cpu().numpy()
        else:
            img = images_remade[index].data.numpy()
        img = np.stack((img, img, img), axis=-1)
        plt.imsave(os.path.join(reconstructed_images_dir, "ckpt_{}_remade_img_{}.png".format(save_iter, j+1)), img, cmap="gray")
        
    # New images

    Z = sample_z(num_gen, k)
    Y = vae.decoder(Z)
    for j in range(num_gen):
        if gpu_is_available:
            img = Y[j].data.cpu().numpy()
        else:
            img = Y[j].data.numpy()
        img = np.stack((img, img, img), axis=-1)
        #img = vutils.make_grid(torch.from_numpy(img), normalize=True, scale_each=True)
        plt.imsave(os.path.join(new_images_dir, "ckpt_{}_new_img_{}.png".format(save_iter, j+1)), img, cmap="gray")
        #writer.add_image('Image', img, save_iter)

    #fig = plt.figure()
    #plt.plot(range(0, save_iter, probability_interval), approximate_distinguishing_probs)
    #fig.suptitle('Gen_Cap {}, Disc_Cap {}'.format(num_hidden_gen, num_hidden_disc), fontsize=20)
    #plt.xlabel('iteration', fontsize=18)
    #plt.ylabel('Distinguishing Probability', fontsize=16)
    #fig.savefig(os.path.join(direct, "dist_prob_gen_{}_disc_{}".format(num_hidden_gen, num_hidden_disc)))
    
    #writer.add_scalar("objective", values.mean(), save_iter)
    #writer.add_scalar("discriminator gradient norms", discrim_grad_norms.mean(), save_iter)
    #writer.add_scalar("generator gradient norms", gen_grad_norms.mean(), save_iter)
    #writer.add_scalar("fake accuracies", fake_accuracies.mean(), save_iter)
    #writer.add_scalar("real accuracies", real_accuracies.mean(), save_iter)



def main(args):
    
    checkpoint = args.checkpoint
    gpu_is_available = False
    if args.gpu > 0:
        gpu_is_available = True
        
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
        
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    mnist_data = torchvision.datasets.MNIST(root, transform=transform, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_data, 
                    batch_size=batch_size, 
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(mnist_data)), 
                                                                                        training_size)))

    vae = vanilla_vae(num_hidden, gpu_is_available)
    
    if gpu_is_available:
        vae = vae.cuda()

    loss_log = []
    mu_log = []
    logvar_log = []

    # Load checkpoint if given
    if checkpoint > 0:
        direct = os.path.join(save_dir, "checkpoint_{}".format(checkpoint))
        vae_ckpt = os.path.join(direct, "vae_ckpt_{}.pkl".format(checkpoint))
        loss_log_ckpt = os.path.join(direct, "loss_log.npy")
        mu_log_ckpt = os.path.join(direct, "mu_log.npy")
        logvar_log_ckpt = os.path.join(direct, "logvar_log.npy")
        vae.load_state_dict(torch.load(vae_ckpt))
        loss_log = np.load(loss_log_ckpt).tolist()
        mu_log = np.load(mu_log_ckpt).tolist()
        logvar_log = np.load(logvar_log_ckpt).tolist()

    vae_optimizer = optim.Adam(vae.parameters(), lr=lr, betas=betas)
    criterion = nn.BCELoss()


    running_value = 0

    # training loop
    for t in range(max_iterations):
        it = t + checkpoint
        
        images, _ = next(iter(mnist_loader)) # get some examples (ignore labels)
        images = Variable(images)

        if gpu_is_available:
            images = images.cuda()

        images_remade, mu, logvar = vae(images)

        vae_loss = loss_function(images_remade, images, mu, logvar)

        if gpu_is_available:
            loss_log.append(vae_loss.data.cpu().numpy())
            mu_log.append(mu.data.cpu().numpy())
            logvar_log.append(logvar.data.cpu().numpy())
        else:
            loss_log.append(vae_loss.data.numpy())
            mu_log.append(mu.data.numpy())
            logvar_log.append(logvar.data.numpy())
        
        running_value += vae_loss.data

        vae.zero_grad()
        vae_loss.backward()
        vae_optimizer.step()

        if (it+1) % print_interval == 0:
            # also check disciminator and generator gradient magnitudes
            #discrim_grad_norm = np.sqrt(np.sum([np.linalg.norm(x.grad.data)**2 for x in discriminator.parameters()]))

            # check how disciminator is doing in terms of accuracy
            #real_accuracy = (x_probs.data.numpy() > 0.5).mean()
            #fake_accuracy = (y_probs.data.numpy() <= 0.5).mean()
            #avg_accuracy = (real_accuracy + fake_accuracy)/2
            #avg_fake_prob = y_probs.data.numpy().mean()
            #avg_real_prob = x_probs.data.numpy().mean()

            #real_accuracies.append(real_accuracy)
            #fake_accuracies.append(fake_accuracy)
            #avg_accuracies.append(avg_accuracy)
            #avg_fake_probs.append(avg_fake_prob)
            #avg_real_probs.append(avg_real_prob)
            #discrim_grad_norms.append(discrim_grad_norm)

            print("######################")
            print("step {}/{}. average loss of last {} steps: {}".format(it+1, max_iterations, print_interval,
                                                                        running_value/print_interval))
            #print("fraction discriminator correct on real: {}, and fake: {}".format(real_accuracy, fake_accuracy))
            #print("average prob on real: {}, and fake: {}".format(avg_real_prob, avg_fake_prob))
            #print("discriminator gradient squared norm: {}".format(discrim_grad_norm))
            running_value = 0


        #if (it+1) % probability_interval == 0:
        #    approximate_distinguishing_probs.append(approximate_distinguishing_prob(discriminator, generator, mnist_data))
        # save ocasionally and save a few sample images generated
        if (it+1) % save_interval == 0:
            save_checkpoint(gpu_is_available, images_remade, vae, num_gen, save_dir, it+1, loss_log, mu_log, logvar_log)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=int, default=0)
    parser.add_argument("--gpu", "-g", type=int, default=0)
    args = parser.parse_args()
    main(args)
