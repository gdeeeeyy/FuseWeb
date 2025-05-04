import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models

# Function to load videos and extract facial landmarks
def load_data(video_dir, label_file, frame_size=(64, 64), num_frames=30):
    videos = []
    labels = []
    
    # Read the labels from the file (assumes label file is a CSV with filename and label)
    with open(label_file, 'r') as f:
        label_map = {line.split(',')[0]: int(line.split(',')[1].strip()) for line in f.readlines()}
    
    # Load video files and extract frames
    for video_file in os.listdir(video_dir):
        video_path = os.path.join(video_dir, video_file)
        label = label_map.get(video_file, None)
        if label is None:
            continue
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        count = 0
        
        while cap.isOpened() and count < num_frames:
            ret, frame = cap.read()
            if not ret:
                break
            # Resize frame
            frame = cv2.resize(frame, frame_size)
            # Pre-process: Detect face and extract facial features here
            # For simplicity, let's just normalize the frames.
            frame = frame / 255.0
            frames.append(frame)
            count += 1
        
        if len(frames) < num_frames:
            continue
        
        videos.append(np.array(frames))
        labels.append(label)
        cap.release()
    
    videos = np.array(videos)
    labels = np.array(labels)
    
    return videos, labels

# Define an RNN (LSTM) model for video sentiment classification
def build_model(input_shape=(30, 64, 64, 3), num_classes=2):
    model = models.Sequential([
        # Input shape: (num_frames, height, width, channels)
        layers.TimeDistributed(layers.Conv2D(32, (3, 3), activation='relu', padding='same'), input_shape=input_shape),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), activation='relu', padding='same')),
        layers.TimeDistributed(layers.MaxPooling2D((2, 2))),
        
        layers.TimeDistributed(layers.Flatten()),  # Flatten the 2D outputs for RNN
        layers.LSTM(128, activation='relu', return_sequences=False),  # LSTM layer to capture temporal features
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')  # Output layer for sentiment classification
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load data
video_dir = 'path_to_video_clips'  # Directory containing video files
label_file = 'path_to_labels_file.csv'  # CSV file containing video filenames and labels

# Load video clips and labels
videos, labels = load_data(video_dir, label_file)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(videos, labels, test_size=0.2, random_state=42)

# Build the model
model = build_model(input_shape=(30, 64, 64, 3), num_classes=2)

# Train the model
history = model.fit(X_train, y_train, epochs=10, batch_size=4, validation_data=(X_val, y_val))

# Evaluate frame-wise accuracy and sequence-level accuracy
def evaluate_framewise_accuracy(model, X, y):
    y_pred = model.predict(X)
    framewise_acc = np.mean(np.argmax(y_pred, axis=-1) == y)
    return framewise_acc

def evaluate_sequence_accuracy(model, X, y):
    y_pred = model.predict(X)
    sequence_acc = np.mean(np.argmax(y_pred, axis=-1) == y)
    return sequence_acc

framewise_acc_train = evaluate_framewise_accuracy(model, X_train, y_train)
framewise_acc_val = evaluate_framewise_accuracy(model, X_val, y_val)
sequence_acc_train = evaluate_sequence_accuracy(model, X_train, y_train)
sequence_acc_val = evaluate_sequence_accuracy(model, X_val, y_val)

print(f"Frame-wise Accuracy (Train): {framewise_acc_train * 100:.2f}%")
print(f"Frame-wise Accuracy (Validation): {framewise_acc_val * 100:.2f}%")
print(f"Sequence-level Accuracy (Train): {sequence_acc_train * 100:.2f}%")
print(f"Sequence-level Accuracy (Validation): {sequence_acc_val * 100:.2f}%")

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
