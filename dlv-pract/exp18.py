import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix

# Define FCN model using VGG16 as backbone
def fcn_vgg16(input_size=(256, 256, 3)):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=input_size)
    
    # Encoder: Use the layers from VGG16 as the backbone
    x = base_model.output
    x = layers.Conv2D(4096, (7, 7), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Conv2D(4096, (1, 1), activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Decoder: Add 1x1 convolutional layer for pixel-wise classification
    x = layers.Conv2D(1, (1, 1), activation='sigmoid')(x)
    
    # Upsample to match input size
    output = layers.UpSampling2D(size=(32, 32), interpolation='bilinear')(x)
    
    model = models.Model(inputs=base_model.input, outputs=output)
    
    # Compile model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Load dataset (assumed to be loaded as numpy arrays)
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

# Build the FCN model with VGG16 backbone
model = fcn_vgg16()

# Train the model
history = model.fit(
    train_images, train_masks,
    validation_data=(val_images, val_masks),
    epochs=10,
    batch_size=8
)

# Evaluate the model
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

# Calculate and visualize metrics

def pixel_accuracy(y_true, y_pred):
    """Pixel Accuracy"""
    correct_pixels = np.sum(np.round(y_true) == np.round(y_pred))
    total_pixels = np.prod(y_true.shape)
    return correct_pixels / total_pixels

def mean_iou(y_true, y_pred):
    """Mean Intersection over Union (IoU)"""
    smooth = 1e-6
    intersection = np.sum(np.round(y_true) * np.round(y_pred))
    union = np.sum(np.round(y_true)) + np.sum(np.round(y_pred)) - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coefficient(y_true, y_pred):
    """Dice Coefficient"""
    smooth = 1e-6
    intersection = np.sum(np.round(y_true) * np.round(y_pred))
    return (2 * intersection + smooth) / (np.sum(np.round(y_true)) + np.sum(np.round(y_pred)) + smooth)

# Test on some samples
index = 5  # Change the index to visualize different samples

# Original image
plt.subplot(1, 3, 1)
plt.imshow(val_images[index])
plt.title('Input Image')

# True mask
plt.subplot(1, 3, 2)
plt.imshow(val_masks[index].squeeze(), cmap='gray')
plt.title('True Mask')

# Predicted mask
predicted_mask = model.predict(val_images[index:index+1])[0]
plt.subplot(1, 3, 3)
plt.imshow(predicted_mask.squeeze(), cmap='gray')
plt.title('Predicted Mask')

plt.tight_layout()
plt.show()

# Calculate metrics for the test image
y_true = val_masks[index]
y_pred = predicted_mask

pixel_acc = pixel_accuracy(y_true, y_pred)
mean_iou_score = mean_iou(y_true, y_pred)
dice_score = dice_coefficient(y_true, y_pred)

print(f"Pixel Accuracy: {pixel_acc * 100:.2f}%")
print(f"Mean IoU: {mean_iou_score:.4f}")
print(f"Dice Coefficient: {dice_score:.4f}")
