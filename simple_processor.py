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
        Applies adaptative threshold with ADAPTIVE_THRESH_GAUSSIAN_C type

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

        Returns:
        --------
        - rotated: dtype('uint8') with single channel
            Image with skew corrected based in best angle found by algorithm
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

        return rotated
    
    def noise_removal(self, image):
        """
        Applies image denoising using Non-local Means Denoising algorithm.

        Parameters:
        -----------
        - image: dtype('uint8') with 3 channels
            Greyscale image
            
        Returns:
        --------
        - dst: dtype('uint8') with single channel
            Denoised image
        """
        img_shape = image.shape
        dst = np.zeros(img_shape)
        return cv2.fastNlMeansDenoising(image, dst, h=5, block_size=7, search_window=21)


    def erode(self, image, kernel_size=5, iterations=1):
        """
        Tries to thin image edges using based on a kernel size using CV2 erode method

        Parameters:
        -----------
        - image: dtype('uint8') with 3 channels
            Greyscale image
        - kernel_size: int, optional, default to 5
            Size of image erode kernel
        - iterations: int, optional, default to 1
            Number of kernel iteractions over the image
            
        Returns:
        --------
        - image: dtype('uint8') with single channel
            Image with thin edges
        """
        kernel = np.zeros((kernel_size, kernel_size), np.uint8)
        return cv2.erode(image, kernel, iterations)


    def preprocess(self, image):
        """
        Executes Simple Processos image correction pipeline in 4 steps:
            1) Binarization;
            2) Noise Removal
            3) Skew Correction
            4) Thinning and Skeletonization

        Parameters:
        -----------
        - image: dtype('uint8') with 3 channels
            Colored Image
            
        Returns:
        --------
        - image: dtype('uint8') with single channel
            Binarized greyscale image with image corrections and noise removal actions applied
        """
        # 1) Binarization
        grayscale = self.get_grayscale(image)
#         thresh = self.thresholding(grayscale)
        adaptative_thresh = self.adaptative_thresholding(thresh)

        # 2) Noise Removal
        denoised = self.noise_removal(adaptative_thresh)

        # 3) Skew Correction
        corrected_skew_image = self.correct_skew(denoised)
        
        # 4) Thinning and Skeletonization
        ts = self.erode(denoised, 2, 3)
        
        return corrected_skew_image
