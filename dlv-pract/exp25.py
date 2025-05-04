import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (change the path to your image)
image = cv2.imread('input_image.jpg')  # Replace with your image path

# Display original image
def show_image(image, title="Image"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Rotation - Rotate by 45 degrees
def rotate_image(image, angle):
    # Get the image dimensions (height, width)
    (h, w) = image.shape[:2]
    # Get the center of the image
    center = (w // 2, h // 2)
    
    # Get the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # Apply the affine transformation (rotation)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    
    return rotated_image

# Scaling - Scale by a factor of 1.5 (increase size)
def scale_image(image, scale_factor):
    # Apply scaling
    scaled_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    return scaled_image

# Translation - Translate by 100 pixels in both x and y directions
def translate_image(image, tx, ty):
    # Define the translation matrix
    translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    # Apply the affine transformation (translation)
    translated_image = cv2.warpAffine(image, translation_matrix, (image.shape[1], image.shape[0]))
    return translated_image

# Apply transformations
rotated_image = rotate_image(image, 45)  # Rotate by 45 degrees
scaled_image = scale_image(image, 1.5)  # Scale by a factor of 1.5
translated_image = translate_image(image, 100, 100)  # Translate by 100 pixels in both x and y

# Show original and transformed images
show_image(image, title="Original Image")
show_image(rotated_image, title="Rotated Image (45°)")
show_image(scaled_image, title="Scaled Image (1.5x)")
show_image(translated_image, title="Translated Image (100, 100)")
