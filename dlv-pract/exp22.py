import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset (you can replace it with a face dataset like CelebA)
(x_train, _), (_, _) = cifar10.load_data()

# Normalize the images
x_train = (x_train - 127.5) / 127.5  # Rescale to [-1, 1]
x_train = x_train.astype(np.float32)

# Define image shape and latent dimension
img_shape = (64, 64, 3)  # 64x64 RGB images
z_dim = 100  # Latent space dimension (random noise vector)

# Build the Generator
def build_generator(z_dim):
    model = models.Sequential()
    model.add(layers.Dense(256 * 8 * 8, activation='relu', input_dim=z_dim))
    model.add(layers.Reshape((8, 8, 256)))
    model.add(layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same', activation='relu'))
    model.add(layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh'))
    return model

# Build the Discriminator
def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, kernel_size=3, strides=2, padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, kernel_size=3, strides=2, padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# Build the DCGAN model (combined generator and discriminator)
def build_gan(generator, discriminator):
    discriminator.trainable = False  # Freeze the discriminator during generator training
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Compile the Discriminator and GAN models
discriminator = build_discriminator(img_shape)
discriminator.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

generator = build_generator(z_dim)
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer='adam')

# Function to train the DCGAN
def train_dcgan(epochs, batch_size, sample_interval):
    # Create folder to save generated images
    os.makedirs('images', exist_ok=True)

    # Adversarial ground truths
    valid = np.ones((batch_size, 1))  # Real labels
    fake = np.zeros((batch_size, 1))  # Fake labels

    # Training loop
    for epoch in range(epochs):
        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]

        # Generate fake images
        z = np.random.normal(0, 1, (batch_size, z_dim))
        gen_imgs = generator.predict(z)

        # Train the discriminator (real images = valid, fake images = fake)
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator (wants discriminator to classify generated images as real)
        g_loss = gan.train_on_batch(z, valid)

        # Print the progress
        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")

        # Save and visualize generated images at regular intervals
        if epoch % sample_interval == 0:
            save_generated_images(epoch)
            plot_generated_images(epoch)

# Function to save generated images
def save_generated_images(epoch, examples=16, dim=(4, 4), figsize=(4, 4)):
    z = np.random.normal(0, 1, (examples, z_dim))
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images to [0, 1]
    
    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig(f"images/{epoch}.png")
    plt.close()

# Function to plot generated images in real-time
def plot_generated_images(epoch, examples=16, dim=(4, 4), figsize=(4, 4)):
    z = np.random.normal(0, 1, (examples, z_dim))
    gen_imgs = generator.predict(z)
    gen_imgs = 0.5 * gen_imgs + 0.5  # Rescale images to [0, 1]
    
    fig, axs = plt.subplots(dim[0], dim[1], figsize=figsize)
    cnt = 0
    for i in range(dim[0]):
        for j in range(dim[1]):
            axs[i, j].imshow(gen_imgs[cnt])
            axs[i, j].axis('off')
            cnt += 1
    plt.show()

# Train the DCGAN
train_dcgan(epochs=5000, batch_size=64, sample_interval=1000)
