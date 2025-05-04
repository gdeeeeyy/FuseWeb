import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (you can replace the image path with your own image)
image = cv2.imread('input_image.jpg')  # Change 'input_image.jpg' to your image path
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Function to apply Canny edge detector and plot results
def apply_canny_and_plot(threshold1, threshold2):
    # Apply Canny edge detector
    edges = cv2.Canny(gray_image, threshold1, threshold2)
    
    # Display the original and edge-detected images
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Canny Edge Detector\nThreshold1: {threshold1}, Threshold2: {threshold2}')
    plt.axis('off')
    
    plt.show()

# Apply Canny edge detection with different thresholds
thresholds = [(50, 150), (100, 200), (150, 250)]

for threshold1, threshold2 in thresholds:
    apply_canny_and_plot(threshold1, threshold2)
