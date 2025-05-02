# Diffusion Model Implementation – Implement a diffusion-based generative model for text or image synthesis.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# --- Hyperparameters ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 28
batch_size = 128
epochs = 5
timesteps = 200
lr = 1e-3

# --- DataLoader (MNIST) ---
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * 2 - 1)])
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST(root="./data", train=True, transform=transform, download=True),
    batch_size=batch_size, shuffle=True
)

# --- Sinusoidal Positional Encoding ---
def sinusoidal_embedding(t, dim):
    half = dim // 2
    emb = np.log(10000) / (half - 1)
    emb = torch.exp(torch.arange(half, device=t.device) * -emb)
    emb = t[:, None] * emb[None, :]
    emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
    return emb

# --- Simple U-Net (Small CNN) ---
class SimpleUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embed = nn.Linear(64, 128)
        self.net = nn.Sequential(
            nn.Conv2d(1 + 1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
        )

    def forward(self, x, t):
        t_emb = sinusoidal_embedding(t, 64)
        t_emb = self.time_embed(t_emb).view(-1, 1, 1, 128)
        t_map = t_emb.repeat(1, 1, x.shape[2], x.shape[3])
        xt = torch.cat([x, t_map.mean(dim=1, keepdim=True)], dim=1)
        return self.net(xt)

# --- Beta Schedule ---
betas = torch.linspace(1e-4, 0.02, timesteps).to(device)
alphas = 1.0 - betas
alpha_cumprod = torch.cumprod(alphas, dim=0)
sqrt_alpha_cumprod = torch.sqrt(alpha_cumprod)
sqrt_1m_alpha_cumprod = torch.sqrt(1 - alpha_cumprod)

# --- Forward Diffusion ---
def forward_diffusion(x0, t):
    noise = torch.randn_like(x0)
    alpha = sqrt_alpha_cumprod[t].view(-1, 1, 1, 1)
    sigma = sqrt_1m_alpha_cumprod[t].view(-1, 1, 1, 1)
    return alpha * x0 + sigma * noise, noise

# --- Initialize Model ---
model = SimpleUNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# --- Training Loop ---
model.train()
for epoch in range(epochs):
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for x, _ in loop:
        x = x.to(device)
        t = torch.randint(0, timesteps, (x.size(0),), device=device)
        x_noisy, noise = forward_diffusion(x, t)
        pred = model(x_noisy, t)
        loss = F.mse_loss(pred, noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())

# --- Sampling (Reverse Diffusion) ---
@torch.no_grad()
def sample(model, img_size, n=16):
    model.eval()
    x = torch.randn(n, 1, img_size, img_size).to(device)
    for t in reversed(range(timesteps)):
        t_batch = torch.full((n,), t, device=device, dtype=torch.long)
        z = torch.randn_like(x) if t > 0 else 0
        alpha_t = alphas[t]
        alpha_bar_t = alpha_cumprod[t]
        sqrt_inv = 1 / torch.sqrt(alpha_t)
        noise_pred = model(x, t_batch)
        x = sqrt_inv * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_bar_t)) * noise_pred) + torch.sqrt(betas[t]) * z
    return x

# --- Generate and Show Images ---
samples = sample(model, image_size)
grid = torchvision.utils.make_grid((samples + 1) / 2, nrow=4)
plt.figure(figsize=(6,6))
plt.imshow(grid.permute(1, 2, 0).cpu())
plt.axis('off')
plt.title("Generated Samples (DDPM)")
plt.show()
