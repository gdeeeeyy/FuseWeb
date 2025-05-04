import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt

# 1. Load the MNIST dataset (image classification)
(X, y), (_, _) = tf.keras.datasets.mnist.load_data()

# Normalize the data (scale pixel values to range [0,1])
X = X / 255.0

# Flatten the images from 28x28 to 784 (1D) for input into ANN
X = X.reshape(X.shape[0], -1)

# Split the dataset into training and validation sets (80% train, 20% test)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 2. Build the Neural Network Model
model = models.Sequential([
    layers.InputLayer(input_shape=(784,)),  # 784 input nodes for the flattened 28x28 image
    layers.Dense(128, activation='relu'),  # Hidden layer with 128 neurons and ReLU activation
    layers.Dense(64, activation='relu'),   # Another hidden layer with 64 neurons
    layers.Dense(10, activation='softmax') # Output layer with 10 neurons (one for each digit class)
])

# 3. Compile the Model
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy', 
              metrics=['accuracy'])

# 4. Train the Model
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 5. Evaluate the Model on Validation Data
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f"Validation Loss: {val_loss}")
print(f"Validation Accuracy: {val_accuracy * 100}%")

# 6. Make Predictions
y_pred = model.predict(X_val)
y_pred = np.argmax(y_pred, axis=1)  # Get the class with the highest probability

# 7. Evaluate with Metrics
print("Classification Report:")
print(classification_report(y_val, y_pred))

# 8. Confusion Matrix
cm = confusion_matrix(y_val, y_pred)
print("Confusion Matrix:")
print(cm)

# 9. Plot Accuracy and Loss curves
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
