# Style Transfer Algorithm – Develop an image transformation model to apply artistic styles to photographs.

import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import matplotlib.pyplot as plt
import copy

# --- Device ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Image Loader ---
loader = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

def load_image(img_path):
    image = Image.open(img_path).convert('RGB')
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

def imshow(tensor, title=None):
    image = tensor.cpu().clone().detach()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

# --- Load Content & Style Images ---
content_img = load_image("content.jpg")  # Replace with your photo path
style_img = load_image("style.jpg")      # Replace with your art style path

# --- Content & Style Loss ---
class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(b * c, h * w)
    G = torch.mm(features, features.t())
    return G.div(b * c * h * w)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, x):
        G = gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

# --- Load VGG and Construct Model ---
cnn = models.vgg19(pretrained=True).features.to(device).eval()
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

class Normalization(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def forward(self, img):
        return (img - self.mean) / self.std

# Layers for content and style
content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

def get_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img):
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    model = nn.Sequential(normalization)

    content_losses = []
    style_losses = []

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_" + name, content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = model(style_img).detach()
            style_loss = StyleLoss(target)
            model.add_module("style_loss_" + name, style_loss)
            style_losses.append(style_loss)

    # Trim layers after last loss
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], (ContentLoss, StyleLoss)):
            break
    model = model[:i+1]

    return model, style_losses, content_losses

# --- Input Image ---
input_img = content_img.clone()

# --- Optimization ---
def run_style_transfer(cnn, norm_mean, norm_std, content_img, style_img, input_img, num_steps=300,
                       style_weight=1e6, content_weight=1):
    model, style_losses, content_losses = get_model_and_losses(
        cnn, norm_mean, norm_std, style_img, content_img)
    optimizer = optim.LBFGS([input_img.requires_grad_()])

    print("Optimizing...")
    run = [0]
    while run[0] <= num_steps:
        def closure():
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = sum(sl.loss for sl in style_losses)
            content_score = sum(cl.loss for cl in content_losses)
            loss = style_weight * style_score + content_weight * content_score
            loss.backward()
            if run[0] % 50 == 0:
                print(f"Step {run[0]}: Style {style_score.item():.2f}, Content {content_score.item():.2f}")
            run[0] += 1
            return loss

        optimizer.step(closure)

    input_img.data.clamp_(0, 1)
    return input_img

# --- Run Style Transfer ---
output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                            content_img, style_img, input_img)

# --- Show Result ---
imshow(output, title="Stylized Output")