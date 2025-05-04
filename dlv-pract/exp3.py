import cv2
import numpy as np
import matplotlib.pyplot as plt

class FeatureExtractor:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)
        if self.image is None:
            raise ValueError("Image not found.")
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

    def edge_detection(self):
        return cv2.Canny(self.gray, 100, 200)

    def corner_detection_harris(self):
        gray_float = np.float32(self.gray)
        harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)
        harris_dilated = cv2.dilate(harris, None)
        corners_image = self.image.copy()
        corners_image[harris_dilated > 0.01 * harris_dilated.max()] = [0, 0, 255]
        return corners_image

    def corner_detection_shi_tomasi(self):
        corners = cv2.goodFeaturesToTrack(self.gray, 100, 0.01, 10)
        corners = np.int0(corners)
        img = self.image.copy()
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (x, y), 4, (0, 255, 0), -1)
        return img

    def keypoints_and_descriptors(self):
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(self.gray, None)
        img_with_kp = cv2.drawKeypoints(self.image, keypoints, None, color=(255, 0, 0), flags=0)
        return img_with_kp, keypoints, descriptors

    def display_all(self):
        edge = self.edge_detection()
        harris = self.corner_detection_harris()
        shi_tomasi = self.corner_detection_shi_tomasi()
        orb_img, kps, desc = self.keypoints_and_descriptors()

        images = [cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB),
                  edge, harris, shi_tomasi, orb_img]
        titles = ["Original", "Edges (Canny)", "Corners (Harris)", "Corners (Shi-Tomasi)", "Keypoints (ORB)"]

        plt.figure(figsize=(15, 8))
        for i in range(5):
            plt.subplot(2, 3, i+1)
            if len(images[i].shape) == 2:
                plt.imshow(images[i], cmap='gray')
            else:
                plt.imshow(images[i])
            plt.title(titles[i])
            plt.axis('off')
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    extractor = FeatureExtractor("sample.jpg")  # Replace with your image path
    extractor.display_all()
