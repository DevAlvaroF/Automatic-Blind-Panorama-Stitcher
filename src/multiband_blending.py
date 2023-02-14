from typing import List, Tuple
import cv2
import numpy as np
from src.utilities import get_new_parameters, single_weights_matrix


#adds the individual weights of an image
# to the global weight matrix
def add_weights(weights_matrix, image, offset):

    H = offset @ image.H
    size, added_offset = get_new_parameters(weights_matrix, image.image, H)

    weights = single_weights_matrix(image.image.shape)
    weights = cv2.warpPerspective(weights, added_offset @ H, size)[:, :, np.newaxis]

    if weights_matrix is None:
        weights_matrix = weights
    else:
        weights_matrix = cv2.warpPerspective(weights_matrix, added_offset, size)

        if len(weights_matrix.shape) == 2:
            weights_matrix = weights_matrix[:, :, np.newaxis]

        weights_matrix = np.concatenate([weights_matrix, weights], axis=2)

    return weights_matrix, added_offset @ offset

# MAximum weight matrix that will be used to compensate for
# each channel intensity
def get_max_weights_matrix(images):

    weights_matrix = None
    offset = np.eye(3)

    for image in images:
        weights_matrix, offset = add_weights(weights_matrix, image, offset)

    weights_maxes = np.max(weights_matrix, axis=2)[:, :, np.newaxis]
    max_weights_matrix = np.where(
        np.logical_and(weights_matrix == weights_maxes, weights_matrix > 0), 1.0, 0.0
    )

    max_weights_matrix = np.transpose(max_weights_matrix, (2, 0, 1))

    return max_weights_matrix, offset

# Transforms a global weight into individual, corresponding to
# the size on the image itself
def get_cropped_weights(images, weights, offset):

    cropped_weights = []
    for i, image in enumerate(images):
        cropped_weights.append(
            cv2.warpPerspective(
                weights[i], np.linalg.inv(offset @ image.H), image.image.shape[:2][::-1]
            )
        )

    return cropped_weights

# For a determined band, creates the corresponding panorama
# Homographies can be estimated from the original ones
def build_band_panorama(images,weights,bands,offset,size):

    pano_weights = np.zeros(size)
    pano_bands = np.zeros((*size, 3))

    for i, image in enumerate(images):
        weights_at_scale = cv2.warpPerspective(weights[i], offset @ image.H, size[::-1])
        pano_weights += weights_at_scale
        pano_bands += weights_at_scale[:, :, np.newaxis] * cv2.warpPerspective(
            bands[i], offset @ image.H, size[::-1]
        )

    return np.divide(
        pano_bands, pano_weights[:, :, np.newaxis], where=pano_weights[:, :, np.newaxis] != 0
    )


# Creates panorama based on number of bands and pre-defined sigma for the pyramids
def multi_band_blending(images, num_bands, sigma):

    max_weights_matrix, offset = get_max_weights_matrix(images)
    size = max_weights_matrix.shape[1:]

    max_weights = get_cropped_weights(images, max_weights_matrix, offset)

    weights = [[cv2.GaussianBlur(max_weights[i], (0, 0), 2 * sigma) for i in range(len(images))]]
    sigma_images = [cv2.GaussianBlur(image.image, (0, 0), sigma) for image in images]
    bands = [
        [
            np.where(
                images[i].image.astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                images[i].image - sigma_images[i],
                0,
            )
            for i in range(len(images))
        ]
    ]

    for k in range(1, num_bands - 1):
        sigma_k = np.sqrt(2 * k + 1) * sigma
        weights.append(
            [cv2.GaussianBlur(weights[-1][i], (0, 0), sigma_k) for i in range(len(images))]
        )

        old_sigma_images = sigma_images

        sigma_images = [
            cv2.GaussianBlur(old_sigma_image, (0, 0), sigma_k)
            for old_sigma_image in old_sigma_images
        ]
        bands.append(
            [
                np.where(
                    old_sigma_images[i].astype(np.int64) - sigma_images[i].astype(np.int64) > 0,
                    old_sigma_images[i] - sigma_images[i],
                    0,
                )
                for i in range(len(images))
            ]
        )

    weights.append([cv2.GaussianBlur(weights[-1][i], (0, 0), sigma_k) for i in range(len(images))])
    bands.append([sigma_images[i] for i in range(len(images))])

    panorama = np.zeros((*max_weights_matrix.shape[1:], 3))

    for k in range(0, num_bands):
        panorama += build_band_panorama(images, weights[k], bands[k], offset, size)
        panorama[panorama < 0] = 0
        panorama[panorama > 255] = 255
        #plt.imshow(panorama)
        #plt.show()
        #P0.displayIm(panorama.astype(np.uint8))
    return panorama
