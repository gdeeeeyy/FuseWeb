import cv2
import numpy as np
import matplotlib.pyplot as plt

class ImageEnhancementToolkit:
    def __init__(self, image_path):
        self.original = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if self.original is None:
            raise ValueError("Image not found or cannot be opened.")
        self.gray = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)

    def show(self, images, titles):
        plt.figure(figsize=(12, 6))
        for i in range(len(images)):
            plt.subplot(2, 3, i+1)
            if len(images[i].shape) == 2:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

    def histogram_equalization(self):
        return cv2.equalizeHist(self.gray)

    def sharpen(self):
        kernel = np.array([[0, -1, 0],
                           [-1, 5,-1],
                           [0, -1, 0]])
        return cv2.filter2D(self.original, -1, kernel)

    def gaussian_blur(self, ksize=(5, 5), sigma=0):
        return cv2.GaussianBlur(self.original, ksize, sigma)

    def sobel_edge_detection(self):
        sobelx = cv2.Sobel(self.gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(self.gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_combined = cv2.magnitude(sobelx, sobely)
        return np.uint8(np.clip(sobel_combined, 0, 255))

    def canny_edge_detection(self, threshold1=100, threshold2=200):
        return cv2.Canny(self.gray, threshold1, threshold2)

    def run_all(self):
        hist_eq = self.histogram_equalization()
        sharpened = self.sharpen()
        blurred = self.gaussian_blur()
        sobel = self.sobel_edge_detection()
        canny = self.canny_edge_detection()

        self.show(
            [self.original, hist_eq, sharpened, blurred, sobel, canny],
            ['Original', 'Histogram Equalized', 'Sharpened', 'Gaussian Blur', 'Sobel Edge', 'Canny Edge']
        )

# Example usage:
if __name__ == "__main__":
    toolkit = ImageEnhancementToolkit("sample.jpg")  # Replace with your image path
    toolkit.run_all()
