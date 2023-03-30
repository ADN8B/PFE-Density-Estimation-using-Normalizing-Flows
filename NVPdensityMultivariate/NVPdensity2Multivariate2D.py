import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde


# Définir les couches d'accouplement
class AffineCoupling(nn.Module):
    def __init__(self, input_dim, hidden_dim, mask):
        super(AffineCoupling, self).__init__()
        self.mask = mask
        self.scale_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
            nn.Tanh()
        )
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

# Définir le Real NVP
class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, base_distribution):
        super(RealNVP, self).__init__()
        self.mask = torch.tensor([i % 2 for i in range(input_dim)])
        self.transforms = nn.ModuleList([AffineCoupling(input_dim, hidden_dim, torch.tensor([np.random.choice([0, 1]) for _ in range(input_dim)])) for _ in range(num_layers)])
        self.base_distribution = base_distribution

    def g(self, z):
        x = z
        for i, transform in reversed(list(enumerate(self.transforms))):
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

# Créer des données d'entraînement à partir d'une distribution gaussienne multivariée
def create_multivariate_gaussian_data(num_samples, means, covs, weights):
    num_gaussians = len(means)
    samples = []
    for _ in range(num_samples):
        gaussian_idx = np.random.choice(num_gaussians, p=weights)
        sample = np.random.multivariate_normal(means[gaussian_idx], covs[gaussian_idx])
        samples.append(sample)

    return torch.tensor(np.array(samples), dtype=torch.float32)

# Paramètres d'entraînement
input_dim = 2
hidden_dim = 1024
num_layers =15
num_epochs = 100
batch_size = 1024
learning_rate = 1e-4

# Création du jeu de données
num_samples = 3000
means = [np.array([3, 3]), np.array([-3, -3])]
covs = [np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]])]
weights = [0.5, 0.5]
train_data = create_multivariate_gaussian_data(num_samples, means, covs, weights)
dataset = TensorDataset(train_data)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialisation du Real NVP, de l'optimiseur et de la fonction de perte
base_distribution = torch.distributions.Normal(0, 1)
real_nvp = RealNVP(input_dim, hidden_dim, num_layers,base_distribution)
optimizer = torch.optim.Adam(real_nvp.parameters(), lr=learning_rate)

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
            optimizer.step()

        train_losses.append(loss.item())
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

    return train_losses

train_losses = train(real_nvp, optimizer, data_loader, num_epochs)


# Test
num_test_samples = 1000
test_data = create_multivariate_gaussian_data(num_samples, means, covs, weights)
test_log_prob = real_nvp.log_prob(test_data)
print(f'Moyenne de la log-vraisemblance sur le jeu de test: {torch.mean(test_log_prob)}')

# Génération d'échantillons
num_generated_samples = 1000
generated_samples = real_nvp.sample(num_generated_samples).detach().numpy()

# Visualisation des données réelles et des échantillons générés
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].scatter(train_data[:, 1], train_data[:, 0],  s=10, alpha=0.5, label="Données réelles")
ax[0].set_title("Données réelles")
ax[0].set_xlabel("x1")
ax[0].set_ylabel("x2")
ax[0].legend()

ax[1].scatter( generated_samples[:, 0],generated_samples[:, 1], s=10, alpha=0.5, color="orange", label="Échantillons générés")
ax[1].set_title("Échantillons générés par Real NVP")
ax[1].set_xlabel("x1")
ax[1].set_ylabel("x2")
ax[1].legend()

plt.show()

def estimate_density(samples):
    kde = gaussian_kde(samples.T)
    return kde
def plot_contour_density(density, samples, grid_size=100, title="Density plot"):
    xmin, ymin = samples.min(axis=0)
    xmax, ymax = samples.max(axis=0)

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



samples = real_nvp.sample(num_generated_samples).detach().numpy()

density = estimate_density(samples)
plot_contour_density(density, samples, title="Real NVP generated samples density plot")


# Définir les paramètres des gaussiennes multivariées
means = np.array([[3, 3], [-3, -3]])
covs = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])
weights = np.array([0.5, 0.5])

# Créer une grille d'échantillonnage
x, y = np.mgrid[-7:7:.1, -7:7:.1]
pos = np.dstack((x, y))

# Calculer la densité de probabilité pour chaque gaussienne
rv1 = multivariate_normal(means[0], covs[0])
rv2 = multivariate_normal(means[1], covs[1])
z1 = rv1.pdf(pos)
z2 = rv2.pdf(pos)

# Mélanger les gaussiennes en utilisant leurs poids
z = weights[0] * z1 + weights[1] * z2

# Créer une figure et un axe
fig = plt.figure()
ax = fig.add_subplot(111)

# Tracer la gaussienne multivariée mixte
ax.contourf(x, y, z)

# Afficher la figure
plt.show()

mean = np.array([[3, 3], [-3, -3]])
cov = np.array([[[1, 0], [0, 1]], [[1, 0], [0, 1]]])

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
    axes[1].scatter(generated_samples[:, 0], generated_samples[:, 1], c='green', alpha=0.5)
    axes[1].set_title("Generated Samples")
    axes[2].scatter(reconstructed_data[:, 0], reconstructed_data[:, 1], c='red', alpha=0.5)
    axes[2].set_title("Reconstructed Data")
    axes[0].set_xlim(-6, 10)
    axes[0].set_ylim(-8, 8)
    axes[1].set_xlim(-4, 10)
    axes[1].set_ylim(-8, 8)
    axes[2].set_xlim(-8, 12)
    axes[2].set_ylim(-8, 8)
    plt.show()

# Reconstruct input data from generated samples
reconstructed_data = real_nvp.g(torch.tensor(samples, dtype=torch.float32)).detach().numpy()

# Visualize original data, generated samples, and reconstructed data
visualize_data(train_data.numpy(), samples, reconstructed_data)