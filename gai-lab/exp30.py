# Conditional GANs (cGANs): Implement a cGAN that can generate specific classes of images based on labeled input. 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Define the Generator class for cGAN
class Generator(nn.Module):
    def __init__(self, z_dim=100, num_classes=10):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Linear(z_dim + num_classes, 256),  # Input z_dim + class labels
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 28 * 28),  # Output image size is 28x28
            nn.Tanh()  # Output values between -1 and 1
        )

    def forward(self, z, labels):
        # Concatenate the noise vector with the class labels
        z = torch.cat([z, labels], dim=1)
        return self.fc(z).view(-1, 1, 28, 28)

# Define the Discriminator class for cGAN
class Discriminator(nn.Module):
    def __init__(self, num_classes=10):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28 + num_classes, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Output between 0 and 1 (real/fake)
        )

    def forward(self, x, labels):
        # Concatenate the image with the class labels
        x = torch.cat([x.view(x.size(0), -1), labels], dim=1)
        return self.fc(x)

# Loss function and optimizers
criterion = nn.BCELoss()
z_dim = 100  # Latent space dimension
num_classes = 10  # Number of classes (digits in MNIST)

# Initialize the models
generator = Generator(z_dim=z_dim, num_classes=num_classes).cuda()
discriminator = Discriminator(num_classes=num_classes).cuda()

# Optimizers
lr = 0.0002
beta1 = 0.5
optimizer_G = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=lr, betas=(beta1, 0.999))

# Data preparation (MNIST dataset)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize between -1 and 1
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Training loop
epochs = 20
real_labels = torch.ones(64, 1).cuda()
fake_labels = torch.zeros(64, 1).cuda()

for epoch in range(epochs):
    for i, (data, labels) in enumerate(train_loader):
        # One-hot encode the labels
        labels_one_hot = torch.zeros(labels.size(0), num_classes).cuda()
        labels_one_hot.scatter_(1, labels.view(-1, 1), 1)
        
        # Train Discriminator
        real_data = data.cuda()
        batch_size = real_data.size(0)

        optimizer_D.zero_grad()
        output_real = discriminator(real_data, labels_one_hot)
        loss_D_real = criterion(output_real, real_labels[:batch_size])
        
        z = torch.randn(batch_size, z_dim).cuda()
        fake_data = generator(z, labels_one_hot)
        output_fake = discriminator(fake_data.detach(), labels_one_hot)  # Detach to avoid updating generator
        loss_D_fake = criterion(output_fake, fake_labels[:batch_size])

        loss_D = loss_D_real + loss_D_fake
        loss_D.backward()
        optimizer_D.step()

        # Train Generator
        optimizer_G.zero_grad()
        output_fake = discriminator(fake_data, labels_one_hot)
        loss_G = criterion(output_fake, real_labels[:batch_size])  # We want to fool the discriminator
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch [{epoch+1}/{epochs}], Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")

    # Generate and save images every 5 epochs
    if (epoch + 1) % 5 == 0:
        with torch.no_grad():
            z = torch.randn(64, z_dim).cuda()
            labels = torch.randint(0, num_classes, (64,)).cuda()
            labels_one_hot = torch.zeros(64, num_classes).cuda()
            labels_one_hot.scatter_(1, labels.view(-1, 1), 1)
            generated_images = generator(z, labels_one_hot).cpu()
            
            fig, axes = plt.subplots(8, 8, figsize=(10, 10))
            for i in range(64):
                ax = axes[i // 8, i % 8]
                ax.imshow(generated_images[i].squeeze(), cmap="gray")
                ax.axis('off')
            plt.tight_layout()
            plt.show()
