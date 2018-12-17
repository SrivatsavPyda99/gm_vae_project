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




class Disciminator(nn.Module):
    def __init__(self, num_hidden):
        super(Disciminator, self).__init__()
        self.flat_img_size = flat_img_size
        self.fc1 = nn.Linear(flat_img_size, 2*num_hidden)
        self.fc2 = nn.Linear(2*num_hidden, num_hidden)
        self.fc3 = nn.Linear(num_hidden, 1)

    def forward(self, x):
        """
        Assumes x is an image
        """
        x = x.view(-1, self.flat_img_size)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = torch.sigmoid(self.fc3(x))
        return x

class Generator(nn.Module):
    def __init__(self, num_hidden):
        super(Generator, self).__init__()
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
        x = torch.tanh(self.fc4(x))
        x = x.view(-1, self.mnist_dim, self.mnist_dim)
        return x

def sample_z(m, k):
    """
    m = number of samples (batch size)
    k = dimension per sample (should probably be around 100)
    returns a numpy array of size m*k of (gaussian) noise to be input to the generator
    """
    return torch.Tensor(np.random.normal(size=(m, k))).cuda()

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

def save_checkpoint(discriminator, generator, num_samples, base_dir, save_iter, values, discrim_grad_norms, gen_grad_norms,
               fake_accuracies, real_accuracies, avg_accuracies, avg_fake_probs, avg_real_probs, approximate_distinguishing_probs, 
               probability_interval, num_hidden_gen, num_hidden_disc):
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
        img = Y[j].data.cpu().numpy()
        img = np.stack((img, img, img), axis=-1)
        #img = vutils.make_grid(torch.from_numpy(img), normalize=True, scale_each=True)
        plt.imsave(os.path.join(direct, "ckpt_{}_img_{}.png".format(save_iter, j+1)), img, cmap="gray")
        #writer.add_image('Image', img, save_iter)

    np.save(os.path.join(direct, "objective.npy"), values)
    np.save(os.path.join(direct, "discrim_grad_norms.npy"), discrim_grad_norms)
    np.save(os.path.join(direct, "gen_grad_norms.npy"), gen_grad_norms)
    np.save(os.path.join(direct, "fake_accuracies.npy"), fake_accuracies)
    np.save(os.path.join(direct, "real_accuracies.npy"), real_accuracies)
    np.save(os.path.join(direct, "approximate_distinguishing_probs.npy"), approximate_distinguishing_probs)

    fig = plt.figure()
    plt.plot(range(0, save_iter, probability_interval), approximate_distinguishing_probs)
    fig.suptitle('Gen_Cap {}, Disc_Cap {}'.format(num_hidden_gen, num_hidden_disc), fontsize=20)
    plt.xlabel('iteration', fontsize=18)
    plt.ylabel('Distinguishing Probability', fontsize=16)
    fig.savefig(os.path.join(direct, "dist_prob_gen_{}_disc_{}".format(num_hidden_gen, num_hidden_disc)))
    
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
    save_interval = 1000
    probability_interval=100
    root = "~/Data/MNIST"
    save_dir = "../checkpoints/gen_{}_disc_{}".format(num_hidden_gen,num_hidden_disc)
    num_gen = 10 # number of samples to generate at save intervals
    gen_steps = 1  # number of generator updates per discriminator update


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])
    mnist_data = torchvision.datasets.MNIST(root, transform=transform, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_data, 
                    batch_size=batch_size, 
                    sampler=torch.utils.data.sampler.SubsetRandomSampler(np.random.choice(range(len(mnist_data)), training_size)))


    generator = Generator(num_hidden_gen).cuda()
    discriminator = Disciminator(num_hidden_disc).cuda()

    objective_values = []
    discrim_grad_norms = []
    gen_grad_norms = []
    fake_accuracies = []
    real_accuracies = []
    avg_accuracies = []
    avg_fake_probs = []
    avg_real_probs = []
    approximate_distinguishing_probs = []

    
    # Load checkpoint if given
    if checkpoint > 0:
        direct = os.path.join(save_dir, "checkpoint_{}".format(checkpoint))
        gen_ckpt = os.path.join(direct, "generator_ckpt_{}.pkl".format(checkpoint))
        discrim_ckpt = os.path.join(direct, "discriminator_ckpt_{}.pkl".format(checkpoint))
        generator.load_state_dict(torch.load(gen_ckpt))
        discriminator.load_state_dict(torch.load(discrim_ckpt))
        approximate_distinguishing_probs = np.load(os.path.join(direct, "approximate_distinguishing_probs.npy")).tolist()

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

            real_accuracies.append(real_accuracy)
            fake_accuracies.append(fake_accuracy)
            avg_accuracies.append(avg_accuracy)
            avg_fake_probs.append(avg_fake_prob)
            avg_real_probs.append(avg_real_prob)
            discrim_grad_norms.append(discrim_grad_norm)

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
                m = torch.min(R[indices==0])
                n = Variable(torch.zeros(R.shape))
                n[indices==1] = m.data[0]
                R = R + n
            
            gen_loss = -torch.log(R).mean()

            generator.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            if (it+1) % print_interval == 0:
                gen_grad_norm = torch.sqrt(torch.sum(torch.Tensor([torch.norm(x.grad.data)**2 for x in generator.parameters()])))
                #gen_grad_norm = np.sqrt(np.sum([np.linalg.norm(x.grad.data)**2 for x in generator.parameters()]))
                gen_grad_norms.append(gen_grad_norm)
                print("generator gradient squared norm: {}".format(gen_grad_norm))

        if (it+1) % probability_interval == 0:
            approximate_distinguishing_probs.append(approximate_distinguishing_prob(discriminator, generator, mnist_data))

        # save ocasionally and save a few sample images generated
        if (it+1) % save_interval == 0:
            save_checkpoint(discriminator, generator, num_gen, save_dir, it+1, objective_values, discrim_grad_norms,
                    gen_grad_norms, fake_accuracies, real_accuracies, avg_accuracies, avg_fake_probs, avg_real_probs, 
                    approximate_distinguishing_probs, probability_interval, num_hidden_gen, num_hidden_disc)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", "-c", type=int, default=0)
    parser.add_argument("--gen_capacity", "-g", type=int, default=256)
    parser.add_argument("--disc_capacity", "-d", type=int, default=256)
    parser.add_argument("--training_size", "-ts", type=int, default=60000)
    parser.add_argument("--batch_size", "-bs", type=int, default=32)
    args = parser.parse_args()
    main(args.checkpoint, args.gen_capacity, args.disc_capacity, args.training_size, args.batch_size)
