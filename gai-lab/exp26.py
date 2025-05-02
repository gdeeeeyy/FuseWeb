# Autoencoder for Image Compression: Train an autoencoder on an image dataset for compression and reconstruction. 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define Autoencoder Architecture
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),  # -> 16x14x14
            nn.ReLU(),
            nn.Conv2d(16, 4, 3, stride=2, padding=1),  # -> 4x7x7
            nn.ReLU()
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 16, 3, stride=2, output_padding=1, padding=1),  # -> 16x14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, output_padding=1, padding=1),  # -> 1x28x28
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Data preparation (MNIST dataset)
transform = transforms.Compose([transforms.ToTensor()])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

# Initialize model, loss, optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Autoencoder().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
n_epochs = 5
for epoch in range(n_epochs):
    for data, _ in train_loader:
        data = data.to(device)
        output = model(data)
        loss = criterion(output, data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Visualize original vs reconstructed images
model.eval()
test_images, _ = next(iter(train_loader))
test_images = test_images.to(device)
with torch.no_grad():
    reconstructed = model(test_images)

# Plot
fig, axs = plt.subplots(2, 6, figsize=(10, 4))
for i in range(6):
    axs[0, i].imshow(test_images[i].cpu().squeeze(), cmap='gray')
    axs[0, i].set_title("Original")
    axs[0, i].axis('off')

    axs[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
    axs[1, i].set_title("Reconstructed")
    axs[1, i].axis('off')

plt.tight_layout()
plt.show()
