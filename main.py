from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from scipy.fftpack import dct, idct
import numpy as np
import os

show_plots = True


def dct_transformation(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def reverse_dct_transformation(a):
    return idct(idct(a, axis=0, norm='ortho'), axis=1, norm='ortho')


def rgb2grayscale(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def save_image(name, file):
    plt.imsave("compressed/" + name, file)


def compress_image(filename, threshold=0):
    threshold = threshold / 1000
    index = 80
    original_image = mpimg.imread(filename)
    split_tup = os.path.splitext(filename)
    gray_scaled_image = rgb2grayscale(original_image)
    plt.gray()

    # 8x8 DCT apply on image
    dctCoefficients = np.zeros(gray_scaled_image.shape)

    for i in range(0, original_image.shape[0], 8):
        for j in range(0, original_image.shape[1], 8):
            dctCoefficients[i:(i + 8), j:(j + 8)] = dct_transformation(gray_scaled_image[i:(i + 8), j:(j + 8)])

    if show_plots:
        plt.figure(figsize=(10, 6))
        plt.subplot(121), plt.imshow(gray_scaled_image[index:index + 8, index:index + 8]), plt.title("8x8 image block",
                                                                                                     size=15)
        plt.subplot(122), plt.imshow(dctCoefficients[index:index + 8, index: index + 8],
                                     vmax=np.max(dctCoefficients) * 0.01,
                                     vmin=0, extent=[0, np.pi, np.pi, 0])
        plt.title("8x8 DCT image block", size=15)
        plt.show()

    dct_threshold = dctCoefficients * (abs(dctCoefficients) > (threshold * np.max(dctCoefficients)))

    im_out = np.zeros(gray_scaled_image.shape)

    for i in range(0, gray_scaled_image.shape[0], 8):
        for j in range(0, gray_scaled_image.shape[1], 8):
            im_out[i:(i + 8), j:(j + 8)] = reverse_dct_transformation(dct_threshold[i:(i + 8), j:(j + 8)])

    if show_plots:
        plt.figure(figsize=(15, 7))
        plt.subplot(121), plt.imshow(gray_scaled_image), plt.axis('off'), plt.title('Original image', size=20)
        plt.subplot(122), plt.imshow(im_out), plt.axis('off'), plt.title(
            "DCT compressed image. Threshold: {}%".format(threshold * 1000), size=20)
        plt.tight_layout()
        plt.show()

    save_name = split_tup[0]
    save_image(save_name + "_compressed_" + str((threshold * 1000)) + ".jpg", im_out)
    print("Image " + save_name + "_compressed_" + str((threshold * 1000)) + ".jpg" + " saved")


# image file, compression percentage
compress_image("img1.jpg", 0)
compress_image("img1.jpg", 1)
compress_image("img1.jpg", 10)
compress_image("img1.jpg", 75)
compress_image("img1.jpg", 100)
compress_image("img1.jpg", 200)
compress_image("gauss.jpg", 0)
compress_image("gauss.jpg", 1)
compress_image("gauss.jpg", 10)
compress_image("gauss.jpg", 50)
compress_image("gauss.jpg", 100)
compress_image("gauss.jpg", 200)
