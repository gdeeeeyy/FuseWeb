# Statistical Evaluation of GANs vs. VAEs – Implement statistical tests to measure the output quality of different generative models.

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from scipy.linalg import sqrtm
from tensorflow.keras.applications.inception_v3 import InceptionV3
from sklearn.metrics import pairwise_distances
import os

# Load the dataset
(x_train, _), (x_test, _) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Add noise to images for the GAN training
def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * np.random.randn(*images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)  # Ensure pixel values are in [0, 1]
    return noisy_images

# Create the GAN model
def build_gan(input_shape):
    # Generator
    generator = models.Sequential()
    generator.add(layers.Dense(128, activation='relu', input_shape=(100,)))
    generator.add(layers.Dense(256, activation='relu'))
    generator.add(layers.Dense(512, activation='relu'))
    generator.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    generator.add(layers.Reshape(input_shape))

    # Discriminator
    discriminator = models.Sequential()
    discriminator.add(layers.Flatten(input_shape=input_shape))
    discriminator.add(layers.Dense(512, activation='relu'))
    discriminator.add(layers.Dense(256, activation='relu'))
    discriminator.add(layers.Dense(1, activation='sigmoid'))

    # GAN Model
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    discriminator.trainable = False
    gan_input = layers.Input(shape=(100,))
    gan_output = discriminator(generator(gan_input))
    gan = models.Model(gan_input, gan_output)
    gan.compile(optimizer='adam', loss='binary_crossentropy')
    
    return generator, discriminator, gan

# Create the VAE model
def build_vae(input_shape):
    # Encoder
    encoder = models.Sequential()
    encoder.add(layers.InputLayer(input_shape=input_shape))
    encoder.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
    encoder.add(layers.MaxPooling2D((2, 2), padding='same'))
    encoder.add(layers.Flatten())
    encoder.add(layers.Dense(256, activation='relu'))
    encoder.add(layers.Dense(128, activation='relu'))
    latent_inputs = layers.Input(shape=(128,))
    
    # Decoder
    decoder = models.Sequential()
    decoder.add(layers.InputLayer(input_shape=(128,)))
    decoder.add(layers.Dense(256, activation='relu'))
    decoder.add(layers.Dense(np.prod(input_shape), activation='sigmoid'))
    decoder.add(layers.Reshape(input_shape))

    vae_model = models.Model(encoder.input, decoder(encoder.output))
    vae_model.compile(optimizer='adam', loss='mse')
    
    return vae_model, encoder, decoder

# Create the InceptionV3 model for evaluating FID and IS
def get_inception_model():
    inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))
    inception_model.trainable = False
    return inception_model

# Function to calculate Inception Score (IS)
def calculate_inception_score(images, model):
    images_resized = np.array([tf.image.resize(image, (299, 299)) for image in images])
    images_resized = np.expand_dims(images_resized, axis=0)
    preds = model.predict(images_resized)
    p_y = np.exp(np.mean(preds, axis=0))
    score = np.exp(np.mean(np.sum(preds * np.log(preds / p_y), axis=1)))
    return score

# Function to calculate Fréchet Inception Distance (FID)
def calculate_fid(real_images, generated_images, model):
    real_images_resized = np.array([tf.image.resize(image, (299, 299)) for image in real_images])
    generated_images_resized = np.array([tf.image.resize(image, (299, 299)) for image in generated_images])
    
    real_features = model.predict(real_images_resized)
    gen_features = model.predict(generated_images_resized)
    
    mu_real, sigma_real = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_gen, sigma_gen = np.mean(gen_features, axis=0), np.cov(gen_features, rowvar=False)
    
    fid = np.sum((mu_real - mu_gen)**2) + np.trace(sigma_real + sigma_gen - 2 * sqrtm(np.dot(sigma_real, sigma_gen)))
    return fid

# Step 1: Train the models (GAN and VAE)
# For simplicity, you can train these models for a few epochs or use pre-trained models.
# Note: In practice, these models need to be trained for many epochs and fine-tuned for better performance.

# Initialize the models
gan_generator, gan_discriminator, gan_model = build_gan(input_shape=(32, 32, 3))
vae_model, vae_encoder, vae_decoder = build_vae(input_shape=(32, 32, 3))

# Initialize InceptionV3 model for FID and IS
inception_model = get_inception_model()

# Step 2: Generate samples from both models (use trained models in practice)
# Generate images using the GAN model
random_latent_vectors = np.random.normal(0, 1, (100, 100))
generated_gan_images = gan_generator.predict(random_latent_vectors)

# Generate images using the VAE model
random_latent_vectors_vae = np.random.normal(0, 1, (100, 128))
generated_vae_images = vae_decoder.predict(random_latent_vectors_vae)

# Step 3: Calculate Inception Score (IS) and FID for both models
# Calculate IS and FID for GAN
gan_is = calculate_inception_score(generated_gan_images, inception_model)
gan_fid = calculate_fid(x_test[:100], generated_gan_images, inception_model)

# Calculate IS and FID for VAE
vae_is = calculate_inception_score(generated_vae_images, inception_model)
vae_fid = calculate_fid(x_test[:100], generated_vae_images, inception_model)

# Step 4: Print the results
print(f"GAN Inception Score: {gan_is}")
print(f"GAN FID: {gan_fid}")
print(f"VAE Inception Score: {vae_is}")
print(f"VAE FID: {vae_fid}")
