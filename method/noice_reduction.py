import cv2
import numpy as np

def noice_reduction(image):
    return cv2.medianBlur(image, 3)  
