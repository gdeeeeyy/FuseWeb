# Generative AI in Healthcare – Create a model for synthetic medical images to improve training datasets for diagnosis.

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, utils
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os

# --- Configuration ---
image_size = 64
channels = 1
latent_dim = 100
batch_size = 64
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Load Medical Dataset ---
# Replace with your own folder path containing medical images (e.g., X-rays)
transform = transforms.Compose([
    transforms.Grayscale(),  # convert to 1 channel
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(root="medical_data", transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# --- Generator ---
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.main(x)

# --- Discriminator ---
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(channels, 128, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# --- Initialize ---
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
fixed_noise = torch.randn(64, latent_dim, 1, 1, device=device)

# --- Training Loop ---
print("Starting Training...")
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        # Train Discriminator
        netD.zero_grad()
        real_images = real_images.to(device)
        b_size = real_images.size(0)
        labels = torch.full((b_size,), 1., device=device)
        output = netD(real_images)
        errD_real = criterion(output, labels)

        noise = torch.randn(b_size, latent_dim, 1, 1, device=device)
        fake_images = netG(noise)
        labels.fill_(0.)
        output = netD(fake_images.detach())
        errD_fake = criterion(output, labels)

        errD = errD_real + errD_fake
        errD.backward()
        optimizerD.step()

        # Train Generator
        netG.zero_grad()
        labels.fill_(1.)  # fool the discriminator
        output = netD(fake_images)
        errG = criterion(output, labels)
        errG.backward()
        optimizerG.step()

    print(f"[{epoch+1}/{epochs}] Loss_D: {errD.item():.4f} Loss_G: {errG.item():.4f}")

    # Save sample images
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake = netG(fixed_noise).detach().cpu()
        utils.save_image(fake, f"generated_epoch_{epoch+1}.png", normalize=True)

print("Training complete. Generated images saved.")
