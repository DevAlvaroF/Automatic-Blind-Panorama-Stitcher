"""
==========================================
        PANORAMA RENDERING BLOCK
==========================================
"""
import numpy as np
import cv2

import src.P0 as P0
import src.config as config
import src.utilities as utilities # blackOut Function to fit surface to Panorama
import src.displayPanoramaBeforeEnhancement as displayPanoramaBeforeEnhancement # Display for Binary Mask
import src.gainCompensation as gainCompensation # Gain Manipulation for intensities
from src.image import Image # Image Objects for easier Manipulation
import src.multiband_blending as multiband_blending # Scale Space Blending

###########################################################################


def draw_bounding_boxes(indices, images,tforms, centerIdx):
    n = np.amax(tforms.shape)
    xlim = np.zeros((n, 2))
    ylim = np.zeros((n, 2))
    hMax = 0
    wMax = 0
    for index in range(len(indices)):
        i = indices[index]
        h = images[i].shape[0]
        w = images[i].shape[1]
        hMax = np.amax([h, hMax])
        wMax = np.amax([w, wMax])
        xlimTmp, ylimTmp = utilities.transformLimits(tforms[0, i], images[i])
        xlim[i, :] = xlimTmp
        ylim[i, :] = ylimTmp

    # Find the minimum and maximum output limits
    # Indices 0 access the min value (returns minimum in the first position the function TransfromLimits)
    # Indices 1 acces the max value
    # indices select the row for the corresponding indices
    xMin = np.amin(xlim[indices, 0])
    xMax = np.amax(xlim[indices, 1])
    yMin = np.amin(ylim[indices, 0])
    yMax = np.amax(ylim[indices, 1])

    # Width and height of panorama + compensation of half widht and
    # half height of central iamge (same offset used for H0)

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
    width = int(width *2* len(images) + images[centerIdx].shape[1] // 2)
    height = int(height *2* len(images) + images[centerIdx].shape[0] // 2)

    # Initialize warped Image, it was being noisy about it
    warpedImage = utilities.setCanvas(images[0], height, width)

    ###################################################
    ###################################################
    ###################################################
    H0 = utilities.axesHomography(images[centerIdx], warpedImage)

    # Create a 2-D spatial reference object defining the size of the panorama
    xLimits = np.array([xMin, xMax])
    yLimits = np.array([yMin, yMax])

    k = len(indices)

    # Plot each image and bounding box
    for index in range(k):
        # Initialize warped Image, it was being noisy about it
        i = indices[index]
        image = images[i]
        # Get size for a grayscale or color image
        if len(image.shape) == 3:
            imRows, imCols, imChannel = image.shape
        else:
            imRows, imCols = image.shape
            imChannel = 1

        #######################################
        # Draw Bounding Box for Planar
        #######################################

        # Get warped image corners in X Y coordinates
        imCorners = np.array([[0, 0], [imCols, 0], [imCols, imRows], [0, imRows], [0, 0]])

        matriz = np.matmul(H0, tforms[0, i])
        # Input as X,Y
        x, y = utilities.forwardAffineTransform(matriz, imCorners[:, 0], imCorners[:, 1])

        xCorrect = np.asarray(x)
        yCorrect = np.asarray(y)

        #######
        #######   This is the perspective change
        warpedImage = cv2.warpPerspective(image, dst=warpedImage, M=matriz, dsize=(width, height),
                                          flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT)

        color = list(np.random.random(size=3) * 256)
        # Line thickness of 2 px
        thickness = 10
        # Draw a rectangle with blue line borders of thickness of 2 px
        for xx in range(len(xCorrect) - 1):
            start_point = (int(xCorrect[xx]), int(yCorrect[xx]))
            end_point = (int(xCorrect[xx + 1]), int(yCorrect[xx + 1]))
            warpedImage = cv2.line(warpedImage, start_point, end_point, color, thickness)

        # This is to PLOT ALL THE PHASES OF THE HOMOGRAPHY PLACING
        if False:
            P0.displayIm(utilities.blackOut(warpedImage), title="Final Plot", factor=5)

    panorama_with_boxes = warpedImage
    return panorama_with_boxes

def renderPanorama(images, tforms, indices, centerIdx, ordering, individualkPoints, individualMatches,
                   individualnumMatches, allCombinedTforms):
    # If only one index detected, the same image will be retrieved
    if len(indices) == 1:
        print('Only 1 image Detected')
        return images[indices[0]]

    # IF the tree has multiple values (a normal Panorama)
    else:
        #############################################
        # DRAW BOUNDING BOX FOR PLANAR
        #############################################

        # Bounding boxes are drawn around each image to show how they were transformed
        if (config.warpType == 'planar') and (config.EXTRA_planarBoundingBox == True):
            panorama_with_boxes = draw_bounding_boxes(indices, images, tforms, centerIdx)
            P0.displayIm(utilities.blackOut(panorama_with_boxes), title="Panorama Bounding Boxes", factor=5)

        #############################################
        # GET BINARY MASK
        #############################################

        # A binary mask is obtained to multiple the final Panorama to reduce
        # Blending variations in the black area
        panorama_for_Binary = displayPanoramaBeforeEnhancement.getPanoBefore(images, tforms, centerIdx, indices)
        threshBin = 0
        maski = cv2.threshold(panorama_for_Binary, threshBin, 255, cv2.THRESH_BINARY)

        # Mask is filled to avoid dark spots
        imgBinary = utilities.flood_fill(utilities.blackOut(maski[1]))
        mask4Pano = (imgBinary.astype(np.uint8)).astype(float) / 255.0

        #############################################
        # GAIN COMPENSATION
        #############################################

        # Gain compensation is done for each channel of an RGB image
        imagesArray = np.empty((len(images), 1), dtype=object)
        for i, ele in enumerate(images):
            imagesArray[i, 0] = ele
        gainImages = gainCompensation.gainCompensation(imagesArray)
        gainImages = [ele for ele in gainImages[0, :]]

        #############################################
        # BAND BLENDING
        #############################################

        # Panorama is blended in scale space to account for intensity variations
        # and visible seems

        objectImages = [Image(i) for i in images]

        k = len(indices)
        for index in range(k):
            i = indices[index]
            objectImages[i].H = tforms[0, i]

        objectImages = ([objectImages[i] for i in indices])
        panoramaBlended = multiband_blending.multi_band_blending(objectImages, config.BLEND_bands, config.BLEND_sigma)

        # Adjust mask based on Blended Pano
        shortRows = np.amin([panoramaBlended.shape[0], mask4Pano.shape[0]])
        shortCols = np.amin([panoramaBlended.shape[1], mask4Pano.shape[1]])
        mask4Pano = mask4Pano[:shortRows, :shortCols, :]
        panoramaBlended = panoramaBlended[:shortRows, :shortCols, :]

        return panoramaBlended, mask4Pano


###########################################################################

# This function recieves all the data previously calculated (in form of array or lists)
# and assigns the values for each subTree (panorama) so that the panorama can
# be created, enhnaced and shown

def displayPanorama(finalPanoramaTforms, images, concomps, centerIdx, allCombinedTforms, subTreesIdx, ordering,
                    kpComplete, allMatches, numMatches):
    print("Building Panorama...")
    # Initialize
    allPanoramas = np.empty((1, len(finalPanoramaTforms)), dtype=object)
    croppedPanoramas = np.empty((1, len(finalPanoramaTforms)), dtype=object)

    # For each subTree (images that match) display the images
    for ele in range(len(subTreesIdx)):

        # Retrieve the 2D Matrices with the indices of the subTree list
        individualIndices = list(subTreesIdx[ele, 0])
        individualnumMatches = numMatches[np.ix_(list(subTreesIdx[ele, 0]), list(subTreesIdx[ele, 0]))]
        individualMatches = allMatches[np.ix_(list(subTreesIdx[ele, 0]), list(subTreesIdx[ele, 0]))]

        # To aid in the process "multithreading" was implemented so values have
        # to be unpacked (OpenCV restriction)
        nn = individualMatches.shape[0]
        for i in range(nn):
            for j in range(nn):
                if (individualnumMatches[j, i] > 0):
                    matchesTmp = individualMatches[j, i]
                    # Get the matching keypoints for each of the images
                    mId_img1 = []
                    mId_img2 = []
                    for mat in matchesTmp:
                        mId_img1.append(mat.queryIdx)
                        mId_img2.append(mat.trainIdx)
                    matchesReal = np.vstack((mId_img1, mId_img2))
                    individualMatches[j, i] = matchesReal

        # Multiple images will come from the process
        # Variable assignment per subTree indices Value
        individualImages = []
        individualkPoints = []
        for ii in individualIndices:
            individualImages.append(images[ii])
            individualkPoints.append(cv2.KeyPoint_convert((kpComplete[ii])))
        individualTforms = finalPanoramaTforms[ele, 0]
        if len(individualTforms.shape) == 1:
            individualTforms = np.expand_dims(individualTforms, axis=0)
        individualAllCombinedTforms = allCombinedTforms[ele, 0]
        individualCenter = centerIdx[ele, 0]
        individualIndices = np.arange(len(subTreesIdx[ele, 0]))
        individualOrdering = ordering[ele, 0]

        # Call Panorama Render with individual parameters
        tmpPan, mask4Pano = renderPanorama(individualImages, individualTforms, individualIndices, individualCenter,
                                           individualOrdering, individualkPoints, individualMatches,
                                           individualnumMatches, individualAllCombinedTforms)
        # Apply Binary Mask to Panorama
        allPanoramas[0, ele] = tmpPan * mask4Pano

        # Display Panorama
        P0.displayIm(utilities.blackOut(np.asarray(tmpPan * mask4Pano, dtype=np.uint8)), title="Panorama", factor=5)

    return allPanoramas
