# -*- coding: utf-8 -*-

import torch
import torch.utils.data as data 
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch import distributions
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde


class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super(AffineCoupling, self).__init__()
        self.mask = mask
        # Transformation "scale" avec un réseau de neurone
        self.scale_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), 
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
        #  Transformation "translation" avec un réseau de neurone
        self.translation_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        x_masked = x * self.mask
        s = self.scale_transform(x_masked) * (1 - self.mask)
        t = self.translation_transform(x_masked) * (1 - self.mask)
        z = x * torch.exp(s) + t
        log_det_J = s.sum(dim=1)
        return z, log_det_J

    def inverse(self, z):
        z_masked = z * self.mask
        s = self.scale_transform(z_masked) * (1 - self.mask)
        t = self.translation_transform(z_masked) * (1 - self.mask)
        x = (z - t) * torch.exp(-s)
        log_det_J = -s.sum(dim=1)
        return x, log_det_J

def create_mixture_of_gaussians_data(num_samples):
    num_gaussians =1
    means = np.array([[1,1]])
    covs = np.array([[[5, -2], [-2, 5]]])
    weights = np.array([1]) 
    samples = []
    for _ in range(num_samples):
        gaussian_idx = np.random.choice(num_gaussians, p=weights)
        sample = np.random.multivariate_normal(means[gaussian_idx], covs[gaussian_idx])
        samples.append(sample)

    return torch.tensor(np.array(samples), dtype=torch.float32)

class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, base_distribution):
        super(RealNVP, self).__init__()
        self.mask = torch.tensor([i % 2 for i in range(input_dim)])
        self.transforms = nn.ModuleList([AffineCoupling(input_dim, hidden_dim, torch.tensor([np.random.choice([0, 1]) for _ in range(input_dim)])) for _ in range(num_layers)])
        self.base_distribution = base_distribution

    def g(self, z):
        x = z
        for i, transform in enumerate(self.transforms):
            x, _ = transform.inverse(x)
        return x

    def f(self, x):
        z = x
        log_det_J_sum = 0
        for transform in self.transforms:
            z, log_det_J = transform(z)
            log_det_J_sum += log_det_J
        return z, log_det_J_sum

    def log_prob(self, x):
        z, log_det_J = self.f(x)
        log_prob = self.base_distribution.log_prob(z).sum(dim=1)
        return log_prob + log_det_J


    def sample(self, batch_size):
        z = self.base_distribution.sample((batch_size, self.mask.size(0)))
        z = z.to(self.mask.device)
        x = self.g(z)
        return x

# Entraînement
def train(real_nvp, optimizer, data_loader, num_epochs):
    train_losses = []

    for epoch in range(num_epochs):
        for batch in data_loader:
            x, = batch
            optimizer.zero_grad()
            log_prob = real_nvp.log_prob(x)
            loss = -torch.mean(log_prob)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(real_nvp.parameters(), max_norm=5)
            optimizer.step()

        train_losses.append(loss.item())
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    return train_losses

def plot_real_and_generated_data(real_data, generated_samples):
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].scatter(real_data[:, 1], real_data[:, 0], s=10, alpha=0.5, label="Données réelles")
    ax[0].set_title("Données réelles")
    ax[0].set_xlabel("x1")
    ax[0].set_ylabel("x2")
    ax[0].legend()

    ax[1].scatter(generated_samples[:, 0], generated_samples[:, 1], s=10, alpha=0.5, color="orange", label="Échantillons générés")
    ax[1].set_title("Échantillons générés par Real NVP")
    ax[1].set_xlabel("x1")
    ax[1].set_ylabel("x2")
    ax[1].legend()

    plt.show()

def plot_multivariate_gaussian_density(mean, cov):
    x, y = np.mgrid[-5:8:.1, -5:8:.1]
    pos = np.dstack((x, y))
    rv = multivariate_normal(mean, cov)
    z = rv.pdf(pos)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.contourf(x, y, z)
    plt.show()

def estimate_density(samples):
    kde = gaussian_kde(samples.T)
    return kde
def plot_contour_density(density, samples, grid_size=100, title="Density plot"):
    x = np.linspace(-7, 7, grid_size)
    y = np.linspace(-7, 7, grid_size)
    X, Y = np.meshgrid(x, y)
    positions = np.vstack([X.ravel(), Y.ravel()])
    Z = np.reshape(density(positions).T, X.shape)

    plt.figure(figsize=(8, 6))
    plt.title(title)
    plt.contourf(X, Y, Z, cmap='viridis')
    plt.colorbar()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()

# Paramètres d'entraînement
input_dim = 2
hidden_dim = 1024
num_layers =10
num_epochs = 100
batch_size = 1024
learning_rate = 1e-4

# Création du jeu de données
num_samples = 10000
train_data = create_mixture_of_gaussians_data(num_samples)
dataset = torch.utils.data.TensorDataset(train_data)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

base_distribution = torch.distributions.Normal(0, 1)
real_nvp = RealNVP(input_dim, hidden_dim, num_layers,base_distribution)
optimizer = torch.optim.Adam(real_nvp.parameters(), lr=learning_rate)

train_losses = train(real_nvp, optimizer, data_loader, num_epochs)

num_test_samples = 1000
test_data = create_mixture_of_gaussians_data(num_test_samples)
test_log_prob = real_nvp.log_prob(test_data)
print(f'Moyenne de la log-vraisemblance sur le jeu de test: {torch.mean(test_log_prob)}')

num_generated_samples = 1000
generated_samples = real_nvp.sample(num_generated_samples).detach().numpy()
plot_real_and_generated_data(train_data, generated_samples)

samples = real_nvp.sample(num_generated_samples).detach().numpy()
density = estimate_density(samples)
plot_contour_density(density, samples, title="Real NVP generated samples density plot")

mean = [1, 1]
cov = [[5, -2], [-2, 5]]
plot_multivariate_gaussian_density(mean, cov)

# Définir les paramètres de la gaussienne multivariée
mean = np.array([1, 1])
cov = np.array([[5, -2], [-2, 5]])


# Calculer les moments statistiques de la gaussienne multivariée et de l'échantillon
gaussian_mean = mean
gaussian_cov = cov
sample_mean = np.mean(samples, axis=0)
sample_cov = np.cov(samples.T)

# Afficher les moments statistiques de la gaussienne multivariée et de l'échantillon
print('Gaussian Mean:', gaussian_mean)
print('Gaussian Covariance:', gaussian_cov)
print('Sample Mean:', sample_mean)
print('Sample Covariance:', sample_cov)

# Calculer la différence entre les moments statistiques de la gaussienne multivariée et de l'échantillon
mean_diff = np.abs(gaussian_mean - sample_mean)
cov_diff = np.abs(gaussian_cov - sample_cov)

# Afficher la différence entre les moments statistiques de la gaussienne multivariée et de l'échantillon
print('Mean Difference:', mean_diff)
print('Covariance Difference:', cov_diff)



def visualize_data(orig_data, generated_samples, reconstructed_data):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(orig_data[:, 0], orig_data[:, 1], c='blue', alpha=0.5)
    axes[0].set_title("Original Data")
    axes[2].scatter(generated_samples[:, 0], generated_samples[:, 1], c='green', alpha=0.5)
    axes[2].set_title("Generated Samples")
    axes[1].scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], c='red', alpha=0.5)
    axes[1].set_title("Reconstructed Data")
    axes[0].set_xlim(-4, 10)
    axes[0].set_ylim(-8, 8)
    axes[1].set_xlim(-4, 10)
    axes[1].set_ylim(-8, 8)
    axes[2].set_xlim(-4, 10)
    axes[2].set_ylim(-8, 8)
    plt.show()

# Reconstruct input data from generated samples
reconstructed_data = real_nvp.g(torch.tensor(samples, dtype=torch.float32)).detach().numpy()

# Visualize original data, generated samples, and reconstructed data
visualize_data(train_data.numpy(), samples, reconstructed_data)

