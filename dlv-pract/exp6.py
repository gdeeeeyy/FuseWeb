import cv2
import numpy as np
import matplotlib.pyplot as plt

def edge_histogram_representation(image_path):
    # Load original image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges using Canny
    edges = cv2.Canny(gray, 100, 200)

    # Compute edge histogram (intensity distribution of edges)
    edge_values = edges.ravel()
    histogram, bins = np.histogram(edge_values, bins=256, range=[0, 256])

    # Plot original image, edge map, and histogram
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(edges, cmap='gray')
    plt.title('Edge Map (Canny)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.plot(histogram, color='black')
    plt.title('Edge Histogram')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    edge_histogram_representation("sample.jpg")  # Replace with your image path
