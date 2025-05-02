# Variational Autoencoder for Image Synthesis: Train a VAE to generate new samples in a chosen domain (e.g., handwritten digits or anime faces). 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt

# Define the Variational Autoencoder (VAE) class
class VAE(nn.Module):
    def __init__(self, latent_dim=20):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, stride=2, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 64x7x7
            nn.ReLU()
        )
        self.fc1 = nn.Linear(64 * 7 * 7, latent_dim)  # Mean of the latent space
        self.fc2 = nn.Linear(64 * 7 * 7, latent_dim)  # Log-variance of the latent space
        self.fc3 = nn.Linear(latent_dim, 64 * 7 * 7)  # Decoder
        self.decoder = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),  # 32x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),  # 1x28x28
            nn.Sigmoid()  # To ensure the output is between 0 and 1
        )

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        mu = self.fc1(x)
        log_var = self.fc2(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = self.fc3(z)
        z = z.view(z.size(0), 64, 7, 7)
        x_reconstructed = self.decoder(z)
        return x_reconstructed

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z)
        return x_reconstructed, mu, log_var

# Loss function: Reconstruction loss + KL divergence
def loss_function(reconstructed_x, x, mu, log_var):
    # Reconstruction loss (Binary Cross-Entropy for MNIST)
    BCE = nn.functional.binary_cross_entropy(reconstructed_x.view(-1, 28 * 28), x.view(-1, 28 * 28), reduction='sum')
    
    # KL Divergence
    # https://arxiv.org/pdf/1312.6114.pdf
    # https://github.com/pytorch/pytorch/issues/35208
    # KL Div = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # In pytorch, log_var is the log of sigma^2
    # So the KL divergence term becomes:
    # 0.5 * sum(1 + log_var - mu^2 - exp(log_var))
    KL_divergence = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KL_divergence

# Data preparation (MNIST dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalization for MNIST
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Initialize the VAE model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE(latent_dim=20).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training the model
epochs = 10
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reconstructed, mu, log_var = model(data)
        loss = loss_function(reconstructed, data, mu, log_var)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss / len(train_loader.dataset):.4f}")

# Generate new images from random latent vectors (sampling from the latent space)
model.eval()
with torch.no_grad():
    z = torch.randn(64, 20).to(device)  # 64 random samples from the latent space
    generated_images = model.decode(z).cpu()

# Plot generated images
fig, axes = plt.subplots(8, 8, figsize=(10, 10))
for i in range(64):
    ax = axes[i // 8, i % 8]
    ax.imshow(generated_images[i].squeeze(), cmap="gray")
    ax.axis('off')
plt.tight_layout()
plt.show()
