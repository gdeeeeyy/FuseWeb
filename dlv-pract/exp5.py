import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image_path):
    # Step 1: Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found.")
    
    # Step 2: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Step 3: Histogram Equalization
    equalized = cv2.equalizeHist(gray)
    
    # Step 4: Plot images and histograms
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Original grayscale image
    axs[0, 0].imshow(gray, cmap='gray')
    axs[0, 0].set_title('Original Grayscale')
    axs[0, 0].axis('off')

    # Histogram of original grayscale
    axs[1, 0].hist(gray.ravel(), bins=256, range=[0,256], color='blue')
    axs[1, 0].set_title('Original Histogram')

    # Equalized image
    axs[0, 1].imshow(equalized, cmap='gray')
    axs[0, 1].set_title('Equalized Image')
    axs[0, 1].axis('off')

    # Histogram of equalized image
    axs[1, 1].hist(equalized.ravel(), bins=256, range=[0,256], color='green')
    axs[1, 1].set_title('Equalized Histogram')

    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    histogram_equalization("low_contrast.jpg")  # Replace with your low-contrast image
