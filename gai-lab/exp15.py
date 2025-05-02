# 15. CycleGAN for Image-to-Image Translation – Implement CycleGAN for tasks like turning horse images into zebras.

# use this maybe - https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix

import os
import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Generator: ResNet-based ---
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)

class Generator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, num_residuals=6):
        super().__init__()
        model = [
            nn.Conv2d(in_channels, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(),
        ]
        for _ in range(num_residuals):
            model.append(ResidualBlock(256))
        model += [
            nn.ConvTranspose2d(256, 128, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, 7, 1, 3),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# --- Discriminator: PatchGAN ---
class Discriminator(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        model = [
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 4, 1, 1)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

# --- Cycle Consistency Loss ---
def cycle_loss(x_real, x_cycled):
    return torch.mean(torch.abs(x_real - x_cycled))

# --- Load Data (Horse2Zebra) ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Download dataset from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
# and extract it into ./data/horse2zebra/trainA and trainB
data_dir = './data/horse2zebra/'
horse_data = ImageFolder(os.path.join(data_dir, 'trainA'), transform=transform)
zebra_data = ImageFolder(os.path.join(data_dir, 'trainB'), transform=transform)
horse_loader = DataLoader(horse_data, batch_size=1, shuffle=True)
zebra_loader = DataLoader(zebra_data, batch_size=1, shuffle=True)

# --- Initialize Models ---
G_H2Z = Generator().to(device)
G_Z2H = Generator().to(device)
D_H = Discriminator().to(device)
D_Z = Discriminator().to(device)

# --- Losses & Optimizers ---
criterion_GAN = nn.MSELoss()
criterion_cycle = nn.L1Loss()

optimizer_G = optim.Adam(itertools.chain(G_H2Z.parameters(), G_Z2H.parameters()), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_H = optim.Adam(D_H.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D_Z = optim.Adam(D_Z.parameters(), lr=0.0002, betas=(0.5, 0.999))

# --- Training Loop ---
epochs = 1  # Increase this for better results
for epoch in range(epochs):
    for i, (horse, zebra) in enumerate(zip(horse_loader, zebra_loader)):
        real_H = horse[0].to(device)
        real_Z = zebra[0].to(device)

        # Labels
        valid = torch.ones((1, 1, 30, 30), requires_grad=False).to(device)
        fake = torch.zeros((1, 1, 30, 30), requires_grad=False).to(device)

        # --- Train Generators ---
        fake_Z = G_H2Z(real_H)
        rec_H = G_Z2H(fake_Z)

        fake_H = G_Z2H(real_Z)
        rec_Z = G_H2Z(fake_H)

        loss_GAN_H2Z = criterion_GAN(D_Z(fake_Z), valid)
        loss_GAN_Z2H = criterion_GAN(D_H(fake_H), valid)

        loss_cycle_H = criterion_cycle(rec_H, real_H)
        loss_cycle_Z = criterion_cycle(rec_Z, real_Z)

        loss_G = loss_GAN_H2Z + loss_GAN_Z2H + 10 * (loss_cycle_H + loss_cycle_Z)

        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

        # --- Train Discriminator H ---
        loss_D_H = (criterion_GAN(D_H(real_H), valid) + criterion_GAN(D_H(fake_H.detach()), fake)) * 0.5
        optimizer_D_H.zero_grad()
        loss_D_H.backward()
        optimizer_D_H.step()

        # --- Train Discriminator Z ---
        loss_D_Z = (criterion_GAN(D_Z(real_Z), valid) + criterion_GAN(D_Z(fake_Z.detach()), fake)) * 0.5
        optimizer_D_Z.zero_grad()
        loss_D_Z.backward()
        optimizer_D_Z.step()

        if i % 50 == 0:
            print(f"Epoch {epoch+1} [{i}] | G Loss: {loss_G.item():.2f} | D_H: {loss_D_H.item():.2f} | D_Z: {loss_D_Z.item():.2f}")

# --- Visualize Sample Translation ---
def imshow(tensor, title=''):
    image = tensor.squeeze(0).detach().cpu()
    image = image * 0.5 + 0.5  # Unnormalize
    plt.imshow(image.permute(1, 2, 0))
    plt.title(title)
    plt.axis('off')
    plt.show()

with torch.no_grad():
    for horse, _ in horse_loader:
        horse = horse.to(device)
        fake_zebra = G_H2Z(horse)
        imshow(horse, "Original Horse")
        imshow(fake_zebra, "Translated Zebra")
        break
