# Denoising Autoencoder: Implement an autoencoder to remove noise from images. 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # 16x14x14
            nn.ReLU(),
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # 8x7x7
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),  # 16x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),  # 1x28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Data preparation (MNIST + noise)
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Add Gaussian noise to images
def add_noise(imgs, noise_factor=0.5):
    noisy_imgs = imgs + noise_factor * torch.randn_like(imgs)
    noisy_imgs = torch.clip(noisy_imgs, 0., 1.)
    return noisy_imgs

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
epochs = 5
for epoch in range(epochs):
    for imgs, _ in train_loader:
        imgs = imgs.to(device)
        noisy_imgs = add_noise(imgs).to(device)
        outputs = model(noisy_imgs)
        loss = criterion(outputs, imgs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Visualize noisy vs clean vs reconstructed
model.eval()
test_imgs, _ = next(iter(train_loader))
test_imgs = test_imgs.to(device)
noisy_imgs = add_noise(test_imgs).to(device)
with torch.no_grad():
    denoised_imgs = model(noisy_imgs)

# Plotting
fig, axs = plt.subplots(3, 6, figsize=(10, 5))
for i in range(6):
    axs[0, i].imshow(test_imgs[i].cpu().squeeze(), cmap="gray")
    axs[0, i].set_title("Original")
    axs[0, i].axis("off")

    axs[1, i].imshow(noisy_imgs[i].cpu().squeeze(), cmap="gray")
    axs[1, i].set_title("Noisy")
    axs[1, i].axis("off")

    axs[2, i].imshow(denoised_imgs[i].cpu().squeeze(), cmap="gray")
    axs[2, i].set_title("Denoised")
    axs[2, i].axis("off")

plt.tight_layout()
plt.show()
