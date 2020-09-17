import pytesseract
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import interpolation as inter


class SimpleProcessor:
    """
    Preprocess images using OpenCV processing methods
    """

    def get_grayscale(self, image):
        """
        Convert image to greyscale. Uses COLOR_BGR2GRAY color space conversion code


        Parameters:
        -----------
        - image: dtype('uint8') with 3 channels
            Image loaded with OpenCV imread method or other similar method

        Returns:
        --------
        - greyscale_image: dtype('uint8') with a single channel
            Greyscale image
        """
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    def thresholding(self, image, threshold=127):
        """
        Applies threshold with THRESH_TRUNC type

        Parameters:
        -----------
        - threshold: int, default  to 127

        Returns:
        --------
        - dst: dtype('uint8') with single channel
            Image with applied threshold
        """
        return cv2.threshold(image, threshold, 255, cv2.THRESH_TRUNC)[1]


    def adaptative_thresholding(self, image, maxValue=255):
        """
        Applies threshold with THRESH_TRUNC type

        Parameters:
        -----------
        - maxValue: int, default  to 255
            

        Returns:
        --------
        - dst: dtype('uint8') with single channel
            Image with applied threshold
        """
        return cv2.adaptiveThreshold(image, maxValue, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 3)
    

    def correct_skew(self, image, delta=1, limit=5):
        """
        Corrects wrong image skew.
        This function is based on Python Skew correction from https://stackoverflow.com/questions/57964634/python-opencv-skew-correction-for-ocr

        Parameters:
        -----------
        - image: dtype('uint8') with 3 channels
            Image loaded with OpenCV imread method or other similar method
        - delta: int, optional
            Possible variation of correction angle
        - limit: int, optional
            A limit for rotation angle
        """
        def determine_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2)
            return histogram, score

        scores = []
        angles = np.arange(-limit, limit + delta, delta)
        for angle in angles:
            histogram, score = determine_score(image, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, best_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, \
                borderMode=cv2.BORDER_REPLICATE)

        return best_angle, rotated


    def erode(self, image, kernel_size=5, iterations=1):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel, iterations)


    def preprocess(self, image):
        # 1) Binarization
        grayscale = self.get_grayscale(image)
        thresh = self.thresholding(grayscale)
        adaptative_thresh = self.adaptative_thresholding(thresh)
        # 2) Skew Correction
        corrected_skew_image = self.correct_skew(adaptative_thresh)
        # 3) Noise Removal
        # 4) Thinning and Skeletonization
        ts = self.erode(corrected_skew_image[1], 1, 3)
        
        return ts
