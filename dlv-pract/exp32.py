import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.applications import VGG16, VGG19
import matplotlib.pyplot as plt

# Load CIFAR-10 dataset (10 classes, 32x32 images)
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to one-hot encoding
train_labels = tf.keras.utils.to_categorical(train_labels, 10)
test_labels = tf.keras.utils.to_categorical(test_labels, 10)

# Define VGG-16 Model
def build_vgg16():
    model = VGG16(weights=None, input_shape=(32, 32, 3), classes=10)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define VGG-19 Model
def build_vgg19():
    model = VGG19(weights=None, input_shape=(32, 32, 3), classes=10)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Build and Train the VGG-16 Model
vgg16_model = build_vgg16()
vgg16_history = vgg16_model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# Build and Train the VGG-19 Model
vgg19_model = build_vgg19()
vgg19_history = vgg19_model.fit(train_images, train_labels, epochs=10, batch_size=64, validation_data=(test_images, test_labels))

# Evaluate VGG-16 Model
vgg16_test_loss, vgg16_test_acc = vgg16_model.evaluate(test_images, test_labels)
print(f"VGG-16 Test Accuracy: {vgg16_test_acc*100:.2f}%")

# Evaluate VGG-19 Model
vgg19_test_loss, vgg19_test_acc = vgg19_model.evaluate(test_images, test_labels)
print(f"VGG-19 Test Accuracy: {vgg19_test_acc*100:.2f}%")

# Plot accuracy and loss curves for VGG-16 and VGG-19
def plot_training_history(history, model_name):
    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title(f'{model_name} Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.show()

# Plot VGG-16 and VGG-19 training history
plot_training_history(vgg16_history, 'VGG-16')
plot_training_history(vgg19_history, 'VGG-19')
