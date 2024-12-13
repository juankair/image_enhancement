import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
from method.sharpening_edge import sharpening_edge
from method.image_brightening import image_brightening
from method.noice_reduction import noice_reduction
from method.texture_enhancement import texture_enhancement
from method.gaussian_filter import gaussian_filter
from data.image_tasks import image_tasks

def save_histogram(image, save_path):
    """Menyimpan histogram dari gambar sebagai file gambar."""
    colors = ('b', 'g', 'r')
    plt.figure()
    for i, color in enumerate(colors):
        histogram = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(histogram, color=color)
        plt.xlim([0, 256])
    plt.title("Histogram Warna RGB")
    plt.xlabel("Intensitas Warna")
    plt.ylabel("Jumlah Pixel")
    plt.savefig(save_path)
    plt.close()
    print(f"Histogram berhasil disimpan di: {save_path}")

def save_image(image, save_path):
    """Menyimpan gambar ke path yang ditentukan."""
    cv2.imwrite(save_path, image)
    print(f"Gambar berhasil disimpan di: {save_path}")

if __name__ == "__main__":
    result_dir = 'result'

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    enhancement_methods = {
        "brightening": image_brightening,
        "sharpening": sharpening_edge,
        "noice_reduction": noice_reduction,
        "texture_enhancement": texture_enhancement,
        "gaussian_filter" : gaussian_filter
    }

    for task in image_tasks:
        image_path = task["path"]

        if not os.path.exists(image_path):
            print(f"File {image_path} tidak ditemukan.")
            continue

        original_image = cv2.imread(image_path)
        file_name = os.path.splitext(os.path.basename(image_path))[0]

        histogram_path_original = os.path.join(result_dir, f"{file_name}_histogram_original.png")
        save_histogram(original_image, histogram_path_original)

        for method_name, method_function in enhancement_methods.items():
            enhanced_image = method_function(original_image)

            enhanced_image_path = os.path.join(result_dir, f"{file_name}_{method_name}.jpg")
            histogram_path_enhanced = os.path.join(result_dir, f"{file_name}_histogram_{method_name}.png")

            save_image(enhanced_image, enhanced_image_path)
            save_histogram(enhanced_image, histogram_path_enhanced)
