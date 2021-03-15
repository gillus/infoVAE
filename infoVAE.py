import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from sklearn.dummy import DummyClassifier
import numpy as np

import matplotlib.pyplot as plt

from itertools import product


import torch
import torchvision.transforms as transforms
import torchvision.datasets as dsets


def get_data(dataset, batch_size):

    # Get MNIST dataset.
    if dataset == 'MNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = datasets.MNIST(root+'mnist/', train='train',
                                download=True, transform=transform)
        print('Dataset len:', len(dataset))

    # Get SVHN dataset.
    elif dataset == 'SVHN':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor()])

        dataset = datasets.SVHN(root+'svhn/', split='train',
                                download=True, transform=transform)
        print('Dataset len:', len(dataset))

    # Get FashionMNIST dataset.
    elif dataset == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()])

        dataset = datasets.FashionMNIST('./FashionMNIST', train='train',
                                download=True, transform=transform)
        print('Dataset len:', len(dataset))

    # Get CelebA dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'CelebA':
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5),
                (0.5, 0.5, 0.5))])

        dataset = datasets.ImageFolder(root=root+'celeba/', transform=transform)
        print('Dataset len:', len(dataset))

    # Get ChestXRay dataset.
    # MUST ALREADY BE DOWNLOADED IN THE APPROPRIATE DIRECTOR DEFINED BY ROOT PATH!
    elif dataset == 'ChestXRay':
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
            ])

        dataset = datasets.ImageFolder(root=root+'chest_xray/train', transform=transform)
        print('Dataset len:', len(dataset))

    elif dataset == 'aloi':
        transform = transforms.Compose([
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor()
            ])

        dataset = datasets.ImageFolder(root=root+'grey4', transform=transform)
        print('Dataset len:', len(dataset))

    # Create dataloader.
    return dataset



#Manages device/graph status/type handling
def from_latent(net, vec):
    with torch.no_grad():
        net.eval()
        return net.decoder( torch.tensor(vec).to(DEVICE).float() ).cpu().detach().numpy().reshape(28,28)

def get_sampling_grid(net, grid):
    base = torch.zeros(LATENT_SIZE-2)
    image = torch.tensor([])
    for vec in grid:
        image = torch.cat([image, torch.from_numpy(from_latent(net, np.hstack([vec, base]) ) ) ] )
    return image.view(64,1,28,28)

def sample_from_latent(net, save = True, location = 'grid.png'):
    grid = list(product(np.linspace(-1,1,8), np.linspace(-1,1,8) ) )
    results = get_sampling_grid(net, grid)
    if save:
        save_image(results, location)
    return results
#
# from torchvision.utils import make_grid
# def show_image_grid(results):
#     img = make_grid(results, nrow = 8)
#     plt.imshow(img.permute(1,2,0) )
#
# sampled_images = sample_from_latent(net, save = False)
#
# show_image_grid(sampled_images)

def show(img):
    '''
        Helper Function to show images, must convert from GPU tensor to
        CPU ndarray first through img.cpu().detach().numpy()
    '''
    plt.figure()
    plt.imshow(img, cmap = 'gray')


def gaussian_kernel(a, b):
    dim1_1, dim1_2 = a.shape[0], b.shape[0]
    depth = a.shape[1]
    a = a.view(dim1_1, 1, depth)
    b = b.view(1, dim1_2, depth)
    a_core = a.expand(dim1_1, dim1_2, depth)
    b_core = b.expand(dim1_1, dim1_2, depth)
    numerator = (a_core - b_core).pow(2).mean(2)/depth
    return torch.exp(-numerator)

def MMD(a, b):
    return gaussian_kernel(a, a).mean() + gaussian_kernel(b, b).mean() - 2*gaussian_kernel(a, b).mean()

def loss_function(pred, true, latent):
    return (pred-true).pow(2).mean(), MMD(torch.randn(200, LATENT_SIZE, requires_grad = False).to(DEVICE), latent)


class Reshape(nn.Module):
    '''
        Used in a nn.Sequential pipeline to reshape on the fly.
    '''

    def __init__(self, *target_shape):
        super().__init__()
        self.target_shape = target_shape

    def forward(self, x):
        return x.view(*self.target_shape)


class MMD_VAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=5, kernel_size=5, padding=2),  # 1*28*28 -> 5*28*28
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5),  # 5*28*28 -> 5*24*24
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=5, out_channels=5, kernel_size=5),  # 5*24*24 -> 5*20*20
            nn.LeakyReLU(),
            Reshape([-1, 5 * 20 * 20]),
            nn.Linear(in_features=5 * 20 * 20, out_features=5 * 12),
            nn.LeakyReLU(),
            nn.Linear(in_features=5 * 12, out_features=LATENT_SIZE)
        )

        self.decoder = nn.Sequential(
            nn.Linear(in_features=LATENT_SIZE, out_features=5 * 12),
            nn.ReLU(),
            nn.Linear(in_features=5 * 12, out_features=24 * 24),
            nn.ReLU(),
            Reshape([-1, 1, 24, 24]),
            nn.ConvTranspose2d(in_channels=1, out_channels=5, kernel_size=3),  # 1*24*24 -> 5*26*26
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=5, out_channels=10, kernel_size=5),  # 5*26*26 -> 10*30*30
            nn.ReLU(),
            nn.Conv2d(in_channels=10, out_channels=1, kernel_size=3),  # 10*30*30 -> 1*28*28
            nn.Sigmoid()
        )

    def forward(self, X):
        if self.training:
            latent = self.encoder(X.to('cuda'))
            return self.decoder(latent), latent
        else:
            return self.decoder(self.encoder(X.to('cuda')))


def train(net, learning_rate, epochs, train_loader, test_loader, optimizer='Adam'):
    if optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    elif optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=learning_rate,
                              momentum=.95, nesterov=True)

    for epoch in range(epochs):
        training_loss = 0
        training_reconstruction_error = 0
        training_mmd = 0

        net.train()
        for batchnum, X in enumerate(train_loader):
            optimizer.zero_grad()

            X = X[0].reshape(-1, 1, 28, 28)
            reconstruction, mu = net(X)
            reconstruction_error, mmd = loss_function(reconstruction, X.cuda(), mu)
            loss = reconstruction_error + mmd
            loss.backward()

            optimizer.step()

            training_reconstruction_error += reconstruction_error
            training_mmd += mmd
            training_loss += loss

        training_reconstruction_error /= (batchnum + 1)
        training_mmd /= (batchnum + 1)
        training_loss /= (batchnum + 1)
        print('Training loss for epoch %i is %.8f, Reconstruction is %.8f, mmd is %.8f' % (
        epoch, training_loss, training_reconstruction_error, training_mmd))

        # Testing loop

        testing_reconstruction_error = 0
        testing_mmd = 0

        with torch.no_grad():
            for batchnum, X in enumerate(test_loader):
                X = X[0].reshape(-1, 1, 28, 28)
                reconstruction, mu = net(X)
                reconstruction_error, mmd = loss_function(reconstruction, X.cuda(), mu)

                testing_reconstruction_error += reconstruction_error
                testing_mmd += mmd

            testing_reconstruction_error /= (batchnum + 1)
            testing_mmd /= (batchnum + 1)
            print('Testing loss for epoch %i is %.8f, Reconstruction is %.8f, mmd is %.8f' % (
            epoch, testing_reconstruction_error + testing_mmd, testing_reconstruction_error, testing_mmd))
        sample_from_latent(net, save = True, location = 'grid'+str(epoch)+'.png')


USE_CUDA = True
DEVICE   = ('cuda' if USE_CUDA else 'cpu')
torch.cuda.empty_cache()
SEED = 123
BATCH_SIZE = 128

LATENT_SIZE = 4

torch.manual_seed(SEED)

datapath = "./"
localpath = '/content'

import os
os.chdir(datapath)


dataset = get_data('FashionMNIST',BATCH_SIZE)

#train = datasets.MNIST('FashionMNIST', train = True , transform = transforms.ToTensor(), download = True)
#test  = datasets.MNIST('FashionMNIST', train = False, transform = transforms.ToTensor(), download = True)

n = len(dataset)  # total number of examples
n_test = int(0.1 * n)  # take ~10% for test
test_data = torch.utils.data.Subset(dataset, range(n_test))  # take first 10%
train_data = torch.utils.data.Subset(dataset, range(n_test, n))  # take the rest

#train = train.float().to(DEVICE)/256
#test  = test.float().to(DEVICE)/256

train_loader = DataLoader(
    dataset = train_data,
    batch_size = BATCH_SIZE,
    shuffle = True
)

test_loader = DataLoader(
    dataset = test_data,
    batch_size = BATCH_SIZE,
    shuffle = True
)


#net = MMD_VAE().to('cuda')
#
#train(net, learning_rate = .0001, epochs = 100, train_loader = train_loader, test_loader = test_loader)
#sample_from_latent(net, save = True, location = 'grid5.png')
#torch.save(net.state_dict(), './infoVAE.sd')
#net.load_state_dict(torch.load('./infoVAE.sd'))

#clf = DummyClassifier().fit(dataset.data.numpy(),np.zeros(60000))
#def calc_alpha_vector(latent_dimensions):
#    for i in
