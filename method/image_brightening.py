import cv2

def image_brightening(image, alpha=1.2, beta=30):
    brightened = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return brightened
