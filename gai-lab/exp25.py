# AI-Generated Video Synthesis – Implement a generative AI model to create short animated sequences from static images.

# # Clone and setup the first-order-model repo
# !git clone https://github.com/AliaksandrSiarohin/first-order-model
# %cd first-order-model
# !pip install -r requirements.txt

# # Download pretrained checkpoint
# !gdown https://drive.google.com/uc?id=1ccNYhiYg_9OlbwaTka3P2qK9iT-4b8N0 -O checkpoints/vox-cpk.pth.tar

# Run animation demo (source image + driving video)
from demo import load_checkpoints, make_animation
from skimage import img_as_ubyte
from skimage.io import imread, imsave
import imageio
import numpy as np
import matplotlib.pyplot as plt
import warnings
import os

warnings.filterwarnings("ignore")

# Load source image and driving video
source_image = imread('path_to_source_image.jpg')  # e.g., avatar.png
driving_video = imageio.mimread('path_to_driving_video.mp4', memtest=False)

# Resize to 256x256
source_image = resize(source_image, (256, 256))[..., :3]
driving_video = [resize(frame, (256, 256))[..., :3] for frame in driving_video]

# Load model
generator, kp_detector = load_checkpoints(config_path='config/vox-256.yaml',
                                          checkpoint_path='checkpoints/vox-cpk.pth.tar')

# Generate animation
predictions = make_animation(source_image, driving_video, generator, kp_detector, relative=True)

# Save result
imageio.mimsave('generated_animation.mp4', [img_as_ubyte(frame) for frame in predictions])
