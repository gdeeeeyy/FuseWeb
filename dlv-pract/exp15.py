import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Dataset path (modify with your dataset directory)
dataset_path = 'path_to_your_dataset'

# Define the directories for train and test datasets
train_dir = os.path.join(dataset_path, 'train')
test_dir = os.path.join(dataset_path, 'test')

# Prepare data augmentation for training
train_datagen_with_augmentation = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,  # Random rotations
    width_shift_range=0.2,  # Random horizontal shifts
    height_shift_range=0.2,  # Random vertical shifts
    shear_range=0.2,  # Random shearing
    zoom_range=0.2,  # Random zoom
    horizontal_flip=True,  # Random flipping
    fill_mode='nearest'  # Fill mode for newly created pixels after transformations
)

train_datagen_without_augmentation = ImageDataGenerator(rescale=1./255)

test_datagen = ImageDataGenerator(rescale=1./255)

# Create generators for both with and without augmentation
train_generator_with_augmentation = train_datagen_with_augmentation.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

train_generator_without_augmentation = train_datagen_without_augmentation.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # No shuffle for testing to keep labels intact
)

# Build the CNN model from scratch
def build_cnn_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(train_generator_with_augmentation.num_classes, activation='softmax')  # Output layer
    ])
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Train the model with data augmentation
model_with_augmentation = build_cnn_model()

history_with_augmentation = model_with_augmentation.fit(
    train_generator_with_augmentation,
    steps_per_epoch=train_generator_with_augmentation.samples // train_generator_with_augmentation.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Train the model without data augmentation
model_without_augmentation = build_cnn_model()

history_without_augmentation = model_without_augmentation.fit(
    train_generator_without_augmentation,
    steps_per_epoch=train_generator_without_augmentation.samples // train_generator_without_augmentation.batch_size,
    epochs=10,
    validation_data=test_generator,
    validation_steps=test_generator.samples // test_generator.batch_size
)

# Evaluate both models on the test set
test_loss_with_augmentation, test_acc_with_augmentation = model_with_augmentation.evaluate(test_generator)
test_loss_without_augmentation, test_acc_without_augmentation = model_without_augmentation.evaluate(test_generator)

print(f"Test Accuracy with Augmentation: {test_acc_with_augmentation * 100:.2f}%")
print(f"Test Accuracy without Augmentation: {test_acc_without_augmentation * 100:.2f}%")

# Plot accuracy and loss curves for both models
plt.figure(figsize=(12, 10))

# Accuracy plot
plt.subplot(2, 2, 1)
plt.plot(history_with_augmentation.history['accuracy'], label='Train Accuracy (Augmented)')
plt.plot(history_with_augmentation.history['val_accuracy'], label='Val Accuracy (Augmented)')
plt.plot(history_without_augmentation.history['accuracy'], label='Train Accuracy (No Augmentation)')
plt.plot(history_without_augmentation.history['val_accuracy'], label='Val Accuracy (No Augmentation)')
plt.title('Accuracy Comparison')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(2, 2, 2)
plt.plot(history_with_augmentation.history['loss'], label='Train Loss (Augmented)')
plt.plot(history_with_augmentation.history['val_loss'], label='Val Loss (Augmented)')
plt.plot(history_without_augmentation.history['loss'], label='Train Loss (No Augmentation)')
plt.plot(history_without_augmentation.history['val_loss'], label='Val Loss (No Augmentation)')
plt.title('Loss Comparison')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
