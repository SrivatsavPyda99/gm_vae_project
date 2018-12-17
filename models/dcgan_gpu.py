"""
By Collin (11/9/18-12/2/18)
"""

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

#torch.set_default_tensor_type('torch.DoubleTensor')


mnist_dim = 28
flat_img_size = mnist_dim*mnist_dim
k = 100 # size of input to generator
#batch_size = 32
mm=0.8
slope_leaky_relu = 0.2



def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Disciminator(nn.Module):
    def __init__(self, layer_multiplier):
        super(Disciminator, self).__init__()
        self.conv1 = nn.Conv2d(1, layer_multiplier, 13, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(layer_multiplier, momentum=mm)
        self.conv2 = nn.Conv2d(layer_multiplier, layer_multiplier * 2, 9, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(layer_multiplier*2, momentum=mm)
        self.conv3 = nn.Conv2d(layer_multiplier*2, layer_multiplier * 4, 5, 1, 0, bias=False)
        self.bn3 = nn.BatchNorm2d(layer_multiplier*4, momentum=mm)
        self.conv4 = nn.Conv2d(layer_multiplier*4, layer_multiplier * 8, 3, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(layer_multiplier*8, momentum=mm)
        self.conv5 = nn.Conv2d(layer_multiplier*8, 1, 2, 1, 0, bias=False)

    def forward(self, x):
        """
        Assumes x is an image
        """
        x = self.bn1(F.leaky_relu(self.conv1(x), slope_leaky_relu))
        x = self.bn2(F.leaky_relu(self.conv2(x), slope_leaky_relu))
        x = self.bn3(F.leaky_relu(self.conv3(x), slope_leaky_relu))
        x = self.bn4(F.leaky_relu(self.conv4(x), slope_leaky_relu))
        x = F.sigmoid(self.conv5(x))
        return x

class Generator(nn.Module):
    def __init__(self, layer_multiplier):
        super(Generator, self).__init__()
        self.conv1 = nn.ConvTranspose2d(k, layer_multiplier * 8, 2, 1, 0, bias=False)
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
        x = F.tanh(self.conv5(x))
        return x

def sample_z(m, k):
    """
    m = number of samples (batch size)
    k = dimension per sample (should probably be around 100)
    returns a numpy array of size m*k of (gaussian) noise to be input to the generator
    """
    return torch.randn(m, k, 1, 1).cuda()

def approximate_distinguishing_prob(discriminator, generator, mnist_data, num_samples=1000):
    temp_loader = torch.utils.data.DataLoader(mnist_data, 
                    batch_size=num_samples, 
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(mnist_data)), num_samples)))

    Z = Variable(sample_z(num_samples, k))
    Y = generator(Z).detach()


    X = Variable(next(iter(temp_loader))[0]).cuda()

    mean_true_prob = discriminator(X).mean().detach().data
    mean_fake_prob = discriminator(Y).mean().detach().data

    return torch.abs(mean_fake_prob - mean_true_prob)

def save_checkpoint(discriminator, generator, num_samples, base_dir, save_iter, values, num_hidden_gen, num_hidden_disc):
    """
    1. Creates a new directory corresponding to the current iteration
    2. saves discriminator and generator parameters
    3. samples num_samples images from the generator
    4. saves "objective" function over time
    5. saves gradient norms over time
    6. saves discriminator accuracies on both true and fake data over time
    """
    
    direct = os.path.join(base_dir, "checkpoint_{}".format(save_iter))
    if not os.path.exists(direct):
        os.mkdir(direct)
    torch.save(generator.state_dict(), os.path.join(direct, "generator_ckpt_{}.pkl".format(save_iter)))
    torch.save(discriminator.state_dict(), os.path.join(direct, "discriminator_ckpt_{}.pkl".format(save_iter)))

    Z = Variable(sample_z(num_samples, k)).cuda()
    Y = generator(Z).detach()
    for j in range(num_samples):
        img = Y[j].data.cpu().numpy().squeeze()
        img = np.stack((img, img, img), axis=-1)
        #img = vutils.make_grid(torch.from_numpy(img), normalize=True, scale_each=True)
        plt.imsave(os.path.join(direct, "ckpt_{}_img_{}.png".format(save_iter, j+1)), img, cmap="gray")
        #writer.add_image('Image', img, save_iter)

    np.save(os.path.join(direct, "objective.npy"), values)
    #np.save(os.path.join(direct, "discrim_grad_norms.npy"), discrim_grad_norms)
    #np.save(os.path.join(direct, "gen_grad_norms.npy"), gen_grad_norms)
    #np.save(os.path.join(direct, "fake_accuracies.npy"), fake_accuracies)
    #np.save(os.path.join(direct, "real_accuracies.npy"), real_accuracies)
    #np.save(os.path.join(direct, "approximate_distinguishing_probs.npy"), approximate_distinguishing_probs)

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

def main(checkpoint, gen_capacity, disc_capacity, training_size, batch_size):
    num_hidden_gen = gen_capacity # For simplicity, keep this the same for each layer for now
    num_hidden_disc = disc_capacity
    max_iterations = 1000000
    discrim_lr = 0.0002
    gen_lr = 0.0002
    betas = (0.5, 0.999)
    print_interval = 100
    save_interval = 10000
    probability_interval=100
    root = "~/Data/MNIST"
    save_dir = "../checkpoints/dc_gen_{}_disc_{}".format(num_hidden_gen,num_hidden_disc)
    num_gen = 10 # number of samples to generate at save intervals
    gen_steps = 1  # number of generator updates per discriminator update


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    mnist_data = torchvision.datasets.MNIST(root, transform=transform, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_data, 
                    batch_size=batch_size, 
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(mnist_data)), training_size)))


    generator = Generator(num_hidden_gen).cuda()
    discriminator = Disciminator(num_hidden_disc).cuda()



    objective_values = []
    #discrim_grad_norms = []
    #gen_grad_norms = []
    #fake_accuracies = []
    #real_accuracies = []
    #avg_accuracies = []
    #avg_fake_probs = []
    #avg_real_probs = []
    #approximate_distinguishing_probs = []
    
    # Load checkpoint if given
    if checkpoint > 0:
        direct = os.path.join(save_dir, "checkpoint_{}".format(checkpoint))
        gen_ckpt = os.path.join(direct, "generator_ckpt_{}.pkl".format(checkpoint))
        discrim_ckpt = os.path.join(direct, "discriminator_ckpt_{}.pkl".format(checkpoint))
        generator.load_state_dict(torch.load(gen_ckpt))
        discriminator.load_state_dict(torch.load(discrim_ckpt))
        #approximate_distinguishing_probs = np.load(os.path.join(direct, "approximate_distinguishing_probs.npy")).tolist()

    discrim_optimizer = optim.Adam(discriminator.parameters(), lr=discrim_lr, betas=betas)
    gen_optimizer = optim.Adam(generator.parameters(), lr=gen_lr, betas=betas)

    
    

    running_value = 0

    # training loop
    for t in range(max_iterations):
        it = t + checkpoint

        # update discriminator
        Z = Variable(sample_z(batch_size, k)) # noise
        Y = generator(Z).detach() # detach so we don't backprop through the generator
        X, _ = next(iter(mnist_loader)) # get some examples (ignore labels)
        X = Variable(X).cuda()
        
        x_probs = discriminator(X)
        y_probs = discriminator(Y)

        # note: the following may be numerically unstable
        discrim_loss = -(1/batch_size)*(torch.log(x_probs).sum() + torch.log(1-y_probs).sum()) # -1 because maximizing

        objective_value = -1*float(discrim_loss)
        objective_values.append(objective_value)
        running_value += objective_value

        discriminator.zero_grad()
        discrim_loss.backward()
        discrim_optimizer.step()

        if (it+1) % print_interval == 0:
            # also check disciminator and generator gradient magnitudes
            discrim_grad_norm = torch.sqrt(torch.sum(torch.Tensor([torch.norm(x.grad.data)**2 for x in discriminator.parameters()])))

            # check how disciminator is doing in terms of accuracy
            x_total = x_probs > 0.5
            x_total = x_total.data.type(torch.FloatTensor)
            y_total = y_probs <= 0.5
            y_total = y_total.data.type(torch.FloatTensor)
            real_accuracy = x_total.mean()
            fake_accuracy = y_total.mean()
            avg_accuracy = (real_accuracy + fake_accuracy)/2
            avg_fake_prob = y_probs.data.mean()
            avg_real_prob = x_probs.data.mean()

            '''
            real_accuracies.append(real_accuracy)
            fake_accuracies.append(fake_accuracy)
            avg_accuracies.append(avg_accuracy)
            avg_fake_probs.append(avg_fake_prob)
            avg_real_probs.append(avg_real_prob)
            discrim_grad_norms.append(discrim_grad_norm)
            '''

            print("######################")
            print("step {}/{}. average value of last {} steps: {}".format(it+1, max_iterations, print_interval,
                                                                        running_value/print_interval))
            print("fraction discriminator correct on real: {}, and fake: {}".format(real_accuracy, fake_accuracy))
            print("average prob on real: {}, and fake: {}".format(avg_real_prob, avg_fake_prob))
            print("discriminator gradient squared norm: {}".format(discrim_grad_norm))
            running_value = 0

        ######  update generator ######
        for _ in range(gen_steps):
            Z = Variable(sample_z(batch_size, k))
            Y = generator(Z)
            
            R = discriminator(Y)
            indices = R==0
            indices = indices.data
            if len(R[indices]) > 0:
                n = Variable(torch.zeros(R.shape))
                n[indices==1] = 1e-8
                n = n.cuda()
                R = R + n
            
            gen_loss = -torch.log(R).mean()

            generator.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            if (it+1) % print_interval == 0:
                gen_grad_norm = torch.sqrt(torch.sum(torch.Tensor([torch.norm(x.grad.data)**2 for x in generator.parameters()])))
                #gen_grad_norms.append(gen_grad_norm)
                print("generator gradient squared norm: {}".format(gen_grad_norm))
        #if (it+1) % probability_interval == 0:
        #    approximate_distinguishing_probs.append(approximate_distinguishing_prob(discriminator, generator, mnist_data))
        # save ocasionally and save a few sample images generated
        if (it+1) % save_interval == 0:
            save_checkpoint(discriminator, generator, num_gen, save_dir, it+1, objective_values, num_hidden_gen, num_hidden_disc)
            objective_values = []
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=int, default=0)
    parser.add_argument("--gen_capacity", "-g", type=int, default=64)
    parser.add_argument("--disc_capacity", "-d", type=int, default=64)
    parser.add_argument("--training_size", "-ts", type=int, default=60000)
    parser.add_argument("--batch_size", "-bs", type=int, default=128)
    args = parser.parse_args()
    main(args.checkpoint, args.gen_capacity, args.disc_capacity, args.training_size, args.batch_size)
