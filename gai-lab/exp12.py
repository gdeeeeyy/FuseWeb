# Denoising Autoencoder – Train an autoencoder to remove noise from images.

import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

# Step 1: Load and preprocess CIFAR-10 dataset
(x_train, _), (x_test, _) = cifar10.load_data()

# Normalize the data to [0, 1] range
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Step 2: Add noise to images
def add_noise(images, noise_factor=0.5):
    noisy_images = images + noise_factor * np.random.randn(*images.shape)
    noisy_images = np.clip(noisy_images, 0., 1.)  # Ensure pixel values are in [0, 1]
    return noisy_images

x_train_noisy = add_noise(x_train)
x_test_noisy = add_noise(x_test)

# Step 3: Build the Denoising Autoencoder model
def build_dae(input_shape):
    model = models.Sequential()
    # Encoder
    model.add(layers.InputLayer(input_shape=input_shape))
    model.add(layers.Conv2D(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.MaxPooling2D((2, 2), padding="same"))
    
    # Decoder
    model.add(layers.Conv2DTranspose(64, (3, 3), activation="relu", padding="same"))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(32, (3, 3), activation="relu", padding="same"))
    model.add(layers.UpSampling2D((2, 2)))
    model.add(layers.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same"))
    
    return model

# Build and compile the model
dae_model = build_dae(x_train.shape[1:])
dae_model.compile(optimizer="adam", loss="mse")

# Step 4: Train the model
dae_model.fit(x_train_noisy, x_train, epochs=10, batch_size=128, validation_data=(x_test_noisy, x_test))

# Step 5: Evaluate the model and visualize results
# Predict denoised images
denoised_images = dae_model.predict(x_test_noisy)

# Visualize some noisy and denoised images
n = 5  # Number of images to display
plt.figure(figsize=(10, 4))
for i in range(n):
    # Noisy images
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test_noisy[i])
    plt.title("Noisy")
    plt.axis("off")

    # Denoised images
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(denoised_images[i])
    plt.title("Denoised")
    plt.axis("off")
plt.show()
