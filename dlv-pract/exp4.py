import cv2
import numpy as np
import matplotlib.pyplot as plt

def detect_edges(image_path):
    # Load image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError("Image not found.")

    # --- Sobel Edge Detection ---
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = np.uint8(np.clip(sobel, 0, 255))

    # --- Canny Edge Detection ---
    canny = cv2.Canny(image, 100, 200)

    # Display images
    titles = ['Original (Grayscale)', 'Sobel Edge Detection', 'Canny Edge Detection']
    images = [image, sobel, canny]

    plt.figure(figsize=(12, 4))
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.tight_layout()
    plt.show()

    # Explanation
    print("\n💡 Difference Between Sobel and Canny:")
    print("- Sobel detects edges based on gradients in horizontal and vertical directions.")
    print("- It is sensitive to noise and detects thicker edges.")
    print("- Canny uses gradient + non-maximum suppression + double thresholding.")
    print("- It gives thinner, cleaner, and more accurate edges.")

# Example usage
if __name__ == "__main__":
    detect_edges("sample.jpg")  # Replace with your image path
