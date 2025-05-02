# Neural Style Transfer: Implement an image style transfer algorithm to transform photographs into various artistic styles.

# pip install torch torchvision matplotlib pillow

import torch
import torch.optim as optim
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load and preprocess images
def load_image(image_path, max_size=400):
    image = Image.open(image_path)
    # Resize image while maintaining aspect ratio
    if max(image.size) > max_size:
        image = transforms.Resize(max_size)(image)
    # Preprocess the image for the neural network
    preprocessor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image = preprocessor(image).unsqueeze(0)
    return image

# Display the image
def imshow(tensor, title=None):
    image = tensor.clone().detach().cpu().squeeze(0)
    image = image.numpy().transpose(1, 2, 0)
    image = image * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Denormalize
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')

# Load VGG19 model for feature extraction
vgg = models.vgg19(pretrained=True).features.eval().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Get the features of the image
def get_features(image, model, layers):
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers:
            features[name] = x
    return features

# Compute content loss (MSE between content features and generated image features)
def content_loss(content, target):
    return F.mse_loss(target, content)

# Compute style loss (Gram matrix comparison between style features and generated image features)
def gram_matrix(tensor):
    batch_size, channels, height, width = tensor.size()
    tensor = tensor.view(channels, height * width)
    gram = torch.mm(tensor, tensor.t())
    return gram

def style_loss(style, target):
    gram_style = gram_matrix(style)
    gram_target = gram_matrix(target)
    return F.mse_loss(gram_target, gram_style)

# Neural Style Transfer
def run_style_transfer(content_img, style_img, num_steps=500, style_weight=1000000, content_weight=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize generated image (copy of content image)
    generated_img = content_img.clone().requires_grad_(True).to(device)

    # Define optimizer
    optimizer = optim.LBFGS([generated_img])

    # Layers to use for content and style loss
    content_layers = ['21']
    style_layers = ['0', '5', '10', '19']

    # Get features from content and style images
    content_features = get_features(content_img, vgg, content_layers)
    style_features = get_features(style_img, vgg, style_layers)

    for step in range(num_steps):
        def closure():
            generated_img.data.clamp_(0, 1)

            optimizer.zero_grad()

            # Get features for generated image
            generated_features = get_features(generated_img, vgg, content_layers + style_layers)

            # Compute content and style loss
            c_loss = 0
            s_loss = 0

            # Content loss
            for content_layer in content_layers:
                c_loss += content_loss(content_features[content_layer], generated_features[content_layer])

            # Style loss
            for style_layer in style_layers:
                s_loss += style_loss(style_features[style_layer], generated_features[style_layer])

            # Total loss
            total_loss = content_weight * c_loss + style_weight * s_loss
            total_loss.backward()

            return total_loss

        optimizer.step(closure)

        if step % 50 == 0:
            print(f"Step [{step}/{num_steps}], Total Loss: {closure().item():.4f}")
            imshow(generated_img, title=f"Generated Image at Step {step}")

    generated_img.data.clamp_(0, 1)
    return generated_img

# Load your content and style images
content_img_path = "path_to_your_content_image.jpg"  # Change to your content image path
style_img_path = "path_to_your_style_image.jpg"  # Change to your style image path

content_img = load_image(content_img_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
style_img = load_image(style_img_path).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Run style transfer
generated_img = run_style_transfer(content_img, style_img)

# Display the final result
imshow(generated_img, title="Final Image")
plt.show()
