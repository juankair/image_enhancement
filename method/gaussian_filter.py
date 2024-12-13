import cv2
import numpy as np

def gaussian_filter(image):
    return cv2.GaussianBlur(image, (5, 5), sigmaX=1.0)

