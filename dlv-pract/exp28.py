import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Histogram Equalization: Improves the contrast of the image.
def histogram_equalization(image):
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Histogram Equalization
    equ = cv2.equalizeHist(image)
    return equ

# 2. Sharpening: Enhances the edges of the image.
def sharpen(image):
    # Define a kernel for sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    
    # Apply the kernel to the image using filter2D
    sharpened_image = cv2.filter2D(image, -1, kernel)
    return sharpened_image

# 3. Smoothing (Gaussian Blur): Reduces noise and detail.
def gaussian_blur(image, ksize=5):
    # Apply Gaussian Blur
    blurred_image = cv2.GaussianBlur(image, (ksize, ksize), 0)
    return blurred_image

# 4. Edge Detection (Sobel): Detects edges in an image.
def sobel_edge_detection(image):
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Compute Sobel edges in both x and y directions
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    
    # Compute the magnitude of the gradient
    sobel_edges = cv2.magnitude(sobel_x, sobel_y)
    return sobel_edges

# 5. Edge Detection (Canny): Detects edges using Canny edge detector.
def canny_edge_detection(image, threshold1=100, threshold2=200):
    # Convert to grayscale if it's not already
    if len(image.shape) > 2:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Canny edge detection
    canny_edges = cv2.Canny(image, threshold1, threshold2)
    return canny_edges

# Function to display images for comparison
def plot_images(original, enhanced_images, titles):
    plt.figure(figsize=(12, 8))

    # Display original image
    plt.subplot(2, 3, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')

    # Display enhanced images
    for i, (image, title) in enumerate(zip(enhanced_images, titles), start=2):
        plt.subplot(2, 3, i)
        plt.imshow(image, cmap='gray')
        plt.title(title)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example: Apply the toolkit on a low-quality image
if __name__ == "__main__":
    # Load an example image (low-quality image or any image you want to enhance)
    image_path = 'low_quality_image.jpg'  # Change this to your image path
    image = cv2.imread(image_path)

    # Apply image enhancement operations
    eq_image = histogram_equalization(image)
    sharpened_image = sharpen(image)
    blurred_image = gaussian_blur(image, ksize=5)
    sobel_image = sobel_edge_detection(image)
    canny_image = canny_edge_detection(image)

    # Display the original and enhanced images
    enhanced_images = [eq_image, sharpened_image, blurred_image, sobel_image, canny_image]
    titles = ['Histogram Equalization', 'Sharpening', 'Gaussian Blur', 'Sobel Edge Detection', 'Canny Edge Detection']
    
    plot_images(image, enhanced_images, titles)
