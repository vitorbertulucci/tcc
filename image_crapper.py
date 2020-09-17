from glob import glob
from PIL import Image, ImageFilter
from skimage import io, util
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import random
import numpy as np
import pytesseract
from pdf2image import convert_from_path
import imgaug as ia
from imgaug import augmenters as iaa
import unidecode

class Crappyfier(self):
    def __init__(self):
        self.alpha = 0.6
        return

    def random_noise(self, shape, color=(55, 55, 55)):
        import random
        import cv2
        # get image shape
        height, width, channels = shape

        # create blank white image
        overlay = 255 * np.ones(shape=[height, width, channels], dtype=np.uint8)

        # randomize crappy shape and position
        ## 0 - rectangle
        ## 1 - line
        ## 2 - circle
        for i in range(3):
            # get randomic shape position
            x1, y1 = [random.choice(range(0, width)), random.choice(range(0, height))]
            x2, y2 = [random.choice(range(0, width)), random.choice(range(0, height))]

            # select random shape
            shape_range = range(0, 2)
            randomic_shape = random.choice(shape_range)
            if randomic_shape == 0:
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            elif randomic_shape == 1:
            cv2.line(overlay, (x1, y1), (x2, y2), color, 80)
            else:
            cv2.circle(overlay, (x1, y1), (x2, y2), color, -1)
        overlay = cv2.blur(overlay, (100, 100))
        return overlay


    def sequential_cv2_noise(self, image):
            from imgaug import augmenters as iaa

        seq = iaa.Sequential([
            iaa.PerspectiveTransform(random_state=1, scale=0.05),
            iaa.Fog(),
            iaa.Affine(rotate=0.01),
            iaa.GammaContrast(3)
        ])
        return seq.augment_image


    def crappy_pipeline(self, image):
        randomic_noised_overlay = self.random_noise(image.shape)
        alpha = 0.6

        noised_image = cv2.addWeighted(image, alpha, randomic_noised_overlay, 1- alpha, 0)

        cv2_noised_image = self.sequential_cv2_noise(noised_image)

        return cv2_noised_image
