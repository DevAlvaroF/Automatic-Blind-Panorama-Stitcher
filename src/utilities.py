# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 19:07:45 2022

@author: alvar
"""

import cv2
import src.P0 as P0
import numpy as np
from skimage.morphology import reconstruction
from typing import List, Tuple



def axesHomography(im,canvas):
  sizeim = im.shape
  sizecanvas = canvas.shape
  H=np.eye(3) #Creates identity matrix of 3x3
  # We creatae a translation matrix to the cavas
  H[0,2]=(sizecanvas[1]//2)-(sizeim[1]//2)
  H[1,2]=(sizecanvas[0]//2)-(sizeim[0]//2)
  return H


def setCanvas(im, rows,cols):
    # Detecting if its RGB or color image with number of channels
    if len(im.shape) == 3:
        tmp = np.zeros((rows, cols, 3), dtype=np.uint8)  # inverted on purpose because opencv rquires this
        return tmp
    else:
        tmp = np.zeros((rows, cols), dtype=np.uint8)
        return tmp


# This is a custom function created to plot depending on the value
# minimum blue and maximum is red (green somewhere in the middle)
def colorSpectrum(sizeValue,minimumValue,maximumValue):
  halfmax = (minimumValue + maximumValue) / 2.0
  #Gets maximum between2  values
  blue = np.maximum(0.0, 255.0 * (1.0 - sizeValue/halfmax))
  red = np.maximum(0.0, 255.0 * (sizeValue / halfmax - 1.0))
  green = 255 - blue - red

  return red,green,blue

# This function display the KeyPoints of an image
def showKP(im,kp,title = 'Keypoints Found',normalizar = True,factor=2,colorMap = False):
  #we need to convert the image to int because float 64 doesn't work well with cv2.drawKeypoints
  im = np.uint8(im)
  # Same image is assigned to an output image
  imout = im

  #We use the same information from the previous code blocks
  #In case we want to draw colors in the objects detected based on the size
  if colorMap:
    #Convert from keypoint to np array
    pts = []
    pts = [ele.size for ele in kp]
    #Extract min and max value
    maxSize = np.amax(pts)
    minSize = np.amin(pts)
    for keypoint in kp:
      red,gree,blue = colorSpectrum(keypoint.size,minSize,maxSize)
      color = (red,gree,blue)
      #DONT FORGET TO ADD THE [] AROUND KEYPOINT, OTHERWISE IT'S NOT THE CORRECT FORMAT THAT OPENCV IS EXPECTING
      #This is because we are now drawing each point individually to change the color
      cv2.drawKeypoints(im,[keypoint],outImage = imout,color=color,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG+cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
  # If we don't want to dray the color will be red
  else:
    cv2.drawKeypoints(im,kp,outImage = imout,color=(255,0,0),flags=cv2.DRAW_MATCHES_FLAGS_DRAW_OVER_OUTIMG+cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS )
  P0.displayIm(imout,title=title,normalizar=normalizar,factor=factor)

  return imout


# This function display the matches between two images
def showMatches(im1,kp1,im2,kp2,matches, N,title='Matches'):
  #we need to convert the image to int because float 64 doesn't work well with cv2.drawKeypoints
  # We do the same with both images
  im1 = np.uint8(im1)
  im2 = np.uint8(im2)

  # Only the first "N" Matches found with the SIFT (or any detector) are shown (We don't want a lot of matches to be shown)
  matches = matches[0:N]
  outImg = []

  #If the matching mode was 'KNN' we use the proper function, if it was bruteforce we use the "generic" openCV function
  # This is done because they're not designed to work together
  outImg = cv2.drawMatchesKnn(im1,kp1,im2,kp2,[matches],outImg=None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
  
  #P0.displayIm(outImg,factor=3,title=title)
  return(outImg)

def transformLimits(tform, image):
    # Returns the coordinates of the
    # new corners of the image after transfromation

    # 0/0---X--->
    #  |
    #  |
    #  Y
    #  |
    #  |
    #  v
    siz = image.shape
    rows = siz[0]
    cols = siz[1]
    # Convert rows and cols intro width (x) and height (Y)
    w = cols
    h = rows

    # Because we have 4 corners
    cornersX = []
    cornersY = []

    corners = [np.float32([[0], [0], [1]]), np.float32([[w], [h], [1]]), np.float32([[w], [0], [1]]),
               np.float32([[0], [h], [1]])]

    for i in range(len(corners)):
      res = np.matmul(tform, corners[i])

      cornersX.append(res[0, 0])
      cornersY.append(res[1, 0])
    newX = [np.amin(cornersX), np.amax(cornersX)]
    newY = [np.amin(cornersY), np.amax(cornersY)]
    return newX, newY

def forwardAffineTransform(T,v1,v2):
    #v1 is for X
    #v2 is for Y
    if (len(v1.shape) == 1) or (len(v2.shape) == 1):
        v1 = np.expand_dims(v1, axis=1)
        v2 = np.expand_dims(v2, axis=1)
    if v1.shape[1] != 1 or v2.shape[1] != 1:
        print('Vectors must be column-shaped!')
        return
    elif v1.shape[0] != v2.shape[0]:
        print('Vectors must be of equal length!')
        return

    it = len(v1)
    result = np.empty((it,2),dtype=float)
    for j in range(it):
        coord = np.array([[int(v1[j,0])],[int(v2[j,0])],[1]])
        newCoord = np.matmul(T,coord)
        # Divide by last element
        newCoord = newCoord / newCoord[2,0]
        result[j,0] = newCoord[0]
        result[j,1] = newCoord[1]

    return result[:,0],result[:,1]


def ind2sub(array_shape, ind):
    ind = int(ind)
    indice = np.unravel_index(ind, array_shape, 'F')
    rows = int(indice[0])
    cols = int(indice[1])

    return rows, cols

def flood_fill(workingArray):
    seed = np.copy(workingArray)
    seed[1:-1, 1:-1] = workingArray.max()
    mask = workingArray
    filled = reconstruction(seed, mask, method='erosion')

    return filled

def fillUpTri(mat,v):
    idxs = []
    n = mat.shape[0] #assuming square matrix
    item = 0
    for col in range(n):
        k = col + 1
        for row in range(k):
            mat[row,col] = v[item,0]
            item = item + 1

    return mat

def matVecNaN2Zero(mat):
    n = mat.shape[0]
    for row in range(n):
        for col in range(n):
            #If its a numpy array
            if hasattr(mat[row,col], "__len__"):
                # if we find NaN
                if np.isnan(mat[row,col]).any():
                    mat[row,col] = np.zeros((mat[row,col].shape))

    return mat

def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h

def inverte(imagem, name):
    imagem = (255-imagem)
    cv2.imwrite(name, imagem)

def convert_to_homogenous_crd(inp, axis=1):
    if isinstance(inp, list):
        inp = np.array(inp)
    r, c = inp.shape
    if axis == 1:
        out = np.concatenate((inp, np.ones((r, 1))), axis=axis)
    else:
        out = np.concatenate((inp, np.ones((1, c))), axis=axis)
    return out

# This function removes the redundant pixels from the canvas
def blackOut(img):
  verticalOffset = 0
  if verticalOffset == 0: #Default Values given by professor
    if len(img.shape)==3:
      im =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
      mask = np.array((im > 0), np.uint8)
      x,y,w,h=cv2.boundingRect(mask)
      return img[y:y+h,x:x+w,:]
    else:
      mask = np.array((img > 0), np.uint8)
      x,y,w,h=cv2.boundingRect(mask)
      return img[y:y+h,x:x+w]

  else: # Offset value exists and we use an offset value if required

    if len(img.shape)==3:
      im =cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
      mask = np.array((im[verticalOffset:-verticalOffset,:] > 0), np.uint8) #Not counting first and last row. This has noise for somereason.
      x,y,w,h=cv2.boundingRect(mask)
      return img[y:y+h,x:x+w,:]
    else:
      mask = np.array((img[verticalOffset:-verticalOffset,:] > 0), np.uint8)
      x,y,w,h=cv2.boundingRect(mask)
      return img[y:y+h,x:x+w]


def toSquareTrans(tform):
    # this when we estimate affine which is returned 2x3
    try:
        lastRow = np.array([[0,0,1]], dtype=np.float64)
        tform = np.vstack((tform,lastRow))
    # if no matche found sometimes error will be returned, we generate
    # an identity matrix for those cases
    except:
        tform = np.eye(3)

    return tform
    
def apply_homography(H, point):

    point = np.asarray([[point[0][0], point[1][0], 1]]).T
    new_point = H @ point
    return new_point[0:2] / new_point[2]


def apply_homography_list(H, points):
    return [apply_homography(H, point) for point in points]

# After an homography it returns the corners via "Forward Transform"
def get_new_corners(image, H):

    top_left = np.asarray([[0, 0]]).T
    top_right = np.asarray([[image.shape[1], 0]]).T
    bottom_left = np.asarray([[0, image.shape[0]]]).T
    bottom_right = np.asarray([[image.shape[1], image.shape[0]]]).T

    return apply_homography_list(H, [top_left, top_right, bottom_left, bottom_right])

# OFfset matrix to ALWAYS have positive values on X and Y
def get_offset(corners):
    top_left, top_right, bottom_left = corners[:3]
    return np.array(
        [
            [1, 0, max(0, -float(min(top_left[0], bottom_left[0])))],
            [0, 1, max(0, -float(min(top_left[1], top_right[1])))],
            [0, 0, 1],
        ],
        np.float32,
    )

# Return the size of the panorama/image transformed that will contain points warped
def get_new_size(corners_images):

    top_right_x = np.max([corners_image[1][0] for corners_image in corners_images])
    bottom_right_x = np.max([corners_images[3][0] for corners_images in corners_images])

    bottom_left_y = np.max([corners_images[2][1] for corners_images in corners_images])
    bottom_right_y = np.max([corners_images[3][1] for corners_images in corners_images])

    width = int(np.ceil(max(bottom_right_x, top_right_x)))
    height = int(np.ceil(max(bottom_right_y, bottom_left_y)))

    width = min(width, 5000)
    height = min(height, 4000)

    return width, height

# Update the size of both the image and the panorama
def get_new_parameters(panorama, image, H):
    corners = get_new_corners(image, H)
    added_offset = get_offset(corners)

    corners_image = get_new_corners(image, added_offset @ H)
    if panorama is None:
        size = get_new_size([corners_image])
    else:
        corners_panorama = get_new_corners(panorama, added_offset)
        size = get_new_size([corners_image, corners_panorama])

    return size, added_offset

# Create 1D Weight evenly spaced with a defined size
def single_weights_array(size):
    if size % 2 == 1:
        return np.concatenate(
            [np.linspace(0, 1, (size + 1) // 2), np.linspace(1, 0, (size + 1) // 2)[1:]]
        )
    else:
        return np.concatenate([np.linspace(0, 1, size // 2), np.linspace(1, 0, size // 2)])


# Create weight of 2D array (Evenly spaced)
def single_weights_matrix(shape: Tuple[int]) -> np.ndarray:

    return (single_weights_array(shape[0])[:, np.newaxis]@ single_weights_array(shape[1])[:, np.newaxis].T)
