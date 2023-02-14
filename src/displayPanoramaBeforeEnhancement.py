
import cv2
import numpy as np
import src.utilities as utilities


def getPanoBefore(images, tforms, centerIdx, indices):
    # To try to save memory from homography errors
    width = 0
    height = 0
    # Finding biggest width and height from all images
    for im in range(len(images)):
        tmpWidth = images[im].shape[1]
        tmpHeight = images[im].shape[0]
        if tmpHeight > height:
            height = tmpHeight
        if tmpWidth > width:
            width = tmpWidth
    width = int(width * len(images) + images[centerIdx].shape[1] // 2)
    height = int(height * len(images) + images[centerIdx].shape[0] // 2)

    # Initialize warped Image, it was being noisy about it
    warpedImage = utilities.setCanvas(images[0], height, width)

    ###################################################
    ###################################################
    ###################################################
    H0 = utilities.axesHomography(images[centerIdx], warpedImage)

    k = len(indices)

    # Plot each image and bounding box
    for index in range(k):

        i = indices[index]
        image = images[i]
        # Get size for a grayscale or color image
        if len(image.shape) == 3:
            imRows, imCols, imChannel = image.shape
        else:
            imRows, imCols = image.shape
            imChannel = 1

        matriz = np.matmul(H0, tforms[0, i])

        warpedImage = cv2.warpPerspective(image, dst=warpedImage, M=matriz, dsize=(width, height),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

    return warpedImage

