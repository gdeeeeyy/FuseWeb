# StyleGAN for Face Generation: Train a StyleGAN model on a dataset like CelebA to create realistic human faces. 

#terminal
# Install the required libraries
# pip install tensorflow==2.6.0
# pip install dnnlib
# pip install legacy
# pip install torch torchvision
# pip install numpy
# pip install pillow

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import cv2
import dnnlib
import legacy
import matplotlib.pyplot as plt

# Step 1: Preprocess CelebA Data
def preprocess_celeba_data(image_dir, output_dir, image_size=(1024, 1024)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(image_dir):
        image_path = os.path.join(image_dir, filename)
        if filename.endswith('.jpg') or filename.endswith('.png'):
            img = Image.open(image_path)
            img = img.resize(image_size, Image.LANCZOS)
            img.save(os.path.join(output_dir, filename))

# Step 2: Convert CelebA Dataset to TFRecords
def convert_to_tfrecords(image_dir, tfrecords_dir):
    # This requires the dataset_tool.py from the StyleGAN2 repository
    os.system(f"python dataset_tool.py create_from_images {tfrecords_dir} {image_dir}")

# Step 3: Train the StyleGAN2 Model
def train_stylegan2(tfrecords_dir, output_dir, gpus=1, batch_size=32, cfg='stylegan2'):
    os.system(f"python train.py --outdir={output_dir} --data={tfrecords_dir} --gpus={gpus} --batch={batch_size} --cfg={cfg}")

# Step 4: Generate Images using Trained StyleGAN Model
def generate_images(model_path, output_dir, truncation_psi=0.7):
    os.system(f"python generate.py --outdir={output_dir} --truncation_psi={truncation_psi} --network={model_path}")

# Step 5: Display Generated Faces
def display_images(image_dir):
    fig, axes = plt.subplots(1, 5, figsize=(20, 10))
    axes = axes.ravel()

    image_files = os.listdir(image_dir)
    for i in range(5):
        img = Image.open(os.path.join(image_dir, image_files[i]))
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()

# Main Execution Flow
def main():
    # Define paths
    image_dir = '/path/to/celeba/images'  # Replace with your CelebA image directory
    output_dir = '/path/to/preprocessed_images'  # Replace with output directory for resized images
    tfrecords_dir = '/path/to/tfrecords'  # Replace with directory to store TFRecords
    trained_model_path = '/path/to/trained_model.pkl'  # Replace with the path to the trained model snapshot
    generated_faces_dir = '/path/to/generated_faces'  # Replace with directory to save generated faces

    # Step 1: Preprocess the CelebA data
    print("Preprocessing CelebA images...")
    preprocess_celeba_data(image_dir, output_dir)

    # Step 2: Convert dataset to TFRecords format
    print("Converting dataset to TFRecords...")
    convert_to_tfrecords(output_dir, tfrecords_dir)

    # Step 3: Train the StyleGAN2 model
    print("Training StyleGAN2...")
    train_stylegan2(tfrecords_dir, output_dir)

    # Step 4: Generate images from the trained model
    print("Generating images...")
    generate_images(trained_model_path, generated_faces_dir)

    # Step 5: Display some generated faces
    print("Displaying generated faces...")
    display_images(generated_faces_dir)

if __name__ == '__main__':
    main()
