import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2
from tensorflow.keras import layers, models

# Define the encoder-decoder CNN model for binary segmentation
def build_model(input_size=(256, 256, 3)):
    inputs = layers.Input(shape=input_size)
    
    # Encoder
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Decoder
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    
    # Output layer with sigmoid activation for binary segmentation
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    model = models.Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Load and preprocess the data (image and mask)
def load_data(image_dir, mask_dir, img_size=(256, 256)):
    images = []
    masks = []
    
    image_files = os.listdir(image_dir)
    mask_files = os.listdir(mask_dir)
    
    for img_file, mask_file in zip(image_files, mask_files):
        img = cv2.imread(os.path.join(image_dir, img_file))
        img = cv2.resize(img, img_size)
        img = img / 255.0  # Normalize
        
        mask = cv2.imread(os.path.join(mask_dir, mask_file), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_size)
        mask = np.expand_dims(mask, axis=-1)  # Add channel dimension
        mask = mask / 255.0  # Normalize
        
        images.append(img)
        masks.append(mask)
    
    images = np.array(images)
    masks = np.array(masks)
    
    return images, masks

# Define paths for dataset
image_dir = 'path_to_your_images'
mask_dir = 'path_to_your_masks'

# Load data
images, masks = load_data(image_dir, mask_dir)

# Split into train and validation sets
train_images = images[:int(0.8*len(images))]
train_masks = masks[:int(0.8*len(masks))]
val_images = images[int(0.8*len(images)):]
val_masks = masks[int(0.8*len(masks)):]

# Build the encoder-decoder model
model = build_model()

# Define callback to visualize predictions after each epoch
class VisualizationCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        # Get a prediction for the first image in the validation set
        pred_mask = self.model.predict(val_images[:1])[0]
        
        # Plot input image, true mask, and predicted mask
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(val_images[0])
        plt.title('Input Image')
        
        plt.subplot(1, 3, 2)
        plt.imshow(val_masks[0].squeeze(), cmap='gray')
        plt.title('True Mask')
        
        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask.squeeze(), cmap='gray')
        plt.title('Predicted Mask')
        
        plt.show()

# Train the model with the visualization callback
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=10,
    batch_size=8,
    callbacks=[VisualizationCallback()]
)

# Evaluate the model on the validation set
val_loss, val_acc = model.evaluate(val_images, val_masks)
print(f"Validation Accuracy: {val_acc * 100:.2f}%")

# Plot training and validation loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
