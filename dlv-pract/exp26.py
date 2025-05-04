import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load an image (replace 'input_image.jpg' with your image path)
image = cv2.imread('input_image.jpg')  # Change to your image path
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Harris Corner Detection
def harris_corner_detection(image, threshold=0.01):
    # Convert to float32
    gray = np.float32(image)
    
    # Apply Harris Corner detection
    corners = cv2.cornerHarris(gray, 2, 3, 0.04)
    
    # Dilate the corners to make them visible
    corners = cv2.dilate(corners, None)
    
    # Apply threshold to get the corners
    image[corners > threshold * corners.max()] = [0, 0, 255]
    
    return image

# Shi-Tomasi Corner Detection
def shi_tomasi_corner_detection(image, max_corners=100, quality_level=0.01, min_distance=10):
    # Convert to grayscale if not already
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect corners using Shi-Tomasi method
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=max_corners, qualityLevel=quality_level, minDistance=min_distance)
    
    # Draw corners on the image
    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(image, (x, y), 3, (0, 0, 255), -1)
    
    return image

# Apply Harris Corner Detection
harris_image = harris_corner_detection(image.copy(), threshold=0.01)

# Apply Shi-Tomasi Corner Detection
shi_tomasi_image = shi_tomasi_corner_detection(image.copy())

# Show the results
plt.figure(figsize=(12, 6))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

# Harris Corner Detection
plt.subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(harris_image, cv2.COLOR_BGR2RGB))
plt.title('Harris Corner Detection')
plt.axis('off')

# Shi-Tomasi Corner Detection
plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(shi_tomasi_image, cv2.COLOR_BGR2RGB))
plt.title('Shi-Tomasi Corner Detection')
plt.axis('off')

plt.show()
