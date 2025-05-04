import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageProcessor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not found.")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def grayscale(self):
        return self.gray

    def smoothing(self):
        return cv2.GaussianBlur(self.gray, (5, 5), 0)

    def edge_detection_sobel(self):
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        return np.uint8(np.clip(sobel_combined, 0, 255))

    def edge_detection_canny(self):
        return cv2.Canny(self.gray, 100, 200)

    def histogram_equalization(self):
        return cv2.equalizeHist(self.gray)

    def display_all(self):
        grayscale = self.grayscale()
        smooth = self.smoothing()
        sobel = self.edge_detection_sobel()
        canny = self.edge_detection_canny()
        hist_eq = self.histogram_equalization()

        images = [cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB),
                  grayscale, smooth, sobel, canny, hist_eq]
        titles = ["Original", "Grayscale", "Smoothed", "Sobel Edge", "Canny Edge", "Histogram Equalization"]

        plt.figure(figsize=(12, 8))
        for i in range(6):
            plt.subplot(2, 3, i+1)
            if len(images[i].shape) == 2:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# Example Usage
if __name__ == "__main__":
    img_path = "sample.jpg"  # Replace with your image
    processor = ImageProcessor(img_path)
    processor.display_all()
