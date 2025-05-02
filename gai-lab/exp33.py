# Multimodal Generative AI: Develop a model that can generate both text and images based on input prompts (e.g., text-to-image generation using CLIP). 

# pip install torch torchvision transformers ftfy
# pip install git+https://github.com/CompVis/taming-transformers.git
# pip install matplotlib pillow

import torch
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
from taming.models.vqgan import VQModel
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Load pre-trained CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Load VQGAN model (you should have a trained model)
vqgan_model = VQModel.from_pretrained("path_to_your_trained_vqgan_model")  # Specify path to VQGAN model

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
vqgan_model.to(device)

# Define image generation function
def generate_image_from_text(prompt: str, num_steps=100, learning_rate=0.1):
    # Preprocess the input text for CLIP
    inputs = clip_processor(text=[prompt], return_tensors="pt", padding=True, truncation=True).to(device)
    
    # Encode the text prompt with CLIP
    with torch.no_grad():
        text_features = clip_model.get_text_features(**inputs)
    
    # Create a random latent vector to start the image generation
    latent = torch.randn(1, 256, 16, 16).to(device)  # Latent dimension for VQGAN (adjust as necessary)
    
    # Optimizer for the latent space
    optimizer = torch.optim.Adam([latent], lr=learning_rate)
    
    for step in range(num_steps):
        optimizer.zero_grad()

        # Decode the latent vector into an image
        generated_image = vqgan_model.decode(latent)

        # Preprocess the generated image for CLIP
        pil_image = transforms.ToPILImage()(generated_image.squeeze(0).cpu())
        clip_input = clip_processor(images=pil_image, return_tensors="pt").to(device)
        
        # Get image features from CLIP
        image_features = clip_model.get_image_features(**clip_input)
        
        # Compute the loss between the text features and the generated image features
        similarity = F.cosine_similarity(text_features, image_features)
        loss = -similarity.mean()  # Minimize the negative cosine similarity

        # Backpropagate the loss
        loss.backward()
        optimizer.step()

        # Print progress
        if step % 10 == 0:
            print(f"Step [{step}/{num_steps}], Loss: {loss.item():.4f}")
        
        # Display the image at intervals
        if step % 50 == 0:
            display_image(generated_image)

    return generated_image

# Display function for images
def display_image(tensor):
    pil_image = transforms.ToPILImage()(tensor.squeeze(0).cpu())
    plt.imshow(pil_image)
    plt.axis('off')
    plt.show()

# Example of generating an image from a text prompt
prompt = "A beautiful sunset over the ocean with mountains in the background"
generated_image = generate_image_from_text(prompt)

# Save final generated image
final_image = transforms.ToPILImage()(generated_image.squeeze(0).cpu())
final_image.save("generated_image.png")
