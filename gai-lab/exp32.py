# Diffusion Models: Implement a denoising diffusion probabilistic model (DDPM) and compare it with GANs for image generation. 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Hyperparameters
image_size = 28  # For MNIST dataset (28x28 images)
batch_size = 64
epochs = 5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the U-Net architecture for DDPM (a common architecture used for DDPMs)
class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(UNet, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Function to add noise (forward diffusion)
def add_noise(x, t, beta_schedule):
    noise = torch.randn_like(x)
    alpha_t = 1 - beta_schedule[t]
    return torch.sqrt(alpha_t) * x + torch.sqrt(1 - alpha_t) * noise

# Function to remove noise (reverse diffusion)
def reverse_diffusion(model, x_t, t, beta_schedule):
    noise_pred = model(x_t)
    alpha_t = 1 - beta_schedule[t]
    return (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize the U-Net model and optimizer
model = UNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Beta schedule (variance for each timestep in the diffusion process)
beta_schedule = torch.linspace(0.0001, 0.02, 1000).to(device)

# Training loop for DDPM
for epoch in range(epochs):
    model.train()
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        # Random timestep
        t = torch.randint(0, 1000, (data.size(0),), device=device)
        
        # Add noise to the image based on the timestep t
        noisy_data = add_noise(data, t, beta_schedule)
        
        # Perform reverse diffusion to remove noise
        optimizer.zero_grad()
        predicted_noise = reverse_diffusion(model, noisy_data, t, beta_schedule)
        
        # Compute loss (MSE loss between predicted and true noise)
        loss = nn.MSELoss()(predicted_noise, data)
        loss.backward()
        optimizer.step()
        
        if batch_idx % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}")

    # Generate and display some images at the end of each epoch
    with torch.no_grad():
        model.eval()
        sample_data = torch.randn(16, 1, image_size, image_size).to(device)
        generated_image = reverse_diffusion(model, sample_data, torch.full((16,), 999, device=device), beta_schedule)
        generated_image = generated_image.cpu().detach().numpy()
        
        plt.figure(figsize=(8, 8))
        for i in range(16):
            plt.subplot(4, 4, i+1)
            plt.imshow(generated_image[i][0], cmap='gray')
            plt.axis('off')
        plt.show()
