import cv2
import numpy as np

def texture_enhancement(image):
    embossing_kernel = np.array([[-2, -1, 0],
                             [-1, 1, 1],
                             [0, 1, 2]])
    return cv2.filter2D(image, ddepth=-1, kernel=embossing_kernel)

