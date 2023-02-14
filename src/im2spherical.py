import numpy as np
import cv2
from src.utilities import blackOut
# Here we have to calculate the projections [X,Y] on a sphere of radius r=f
# of the rectangular mesh of the images
# Start by discretizing the azimuth and elevation ranges.
# Calculate the ranges using image size and focal length.
# Then compute the coordinates on the sphere for each couple of angle-values
# return such coordinates
# Here we converte image coordinates to spherical coordinates
def spherical_lookupTable(f, im):
    heightIm = im.shape[0]  # Numbero f rows is the height
    widthIm = im.shape[1]

    # Get Image Size
    ydim = heightIm
    xdim = widthIm

    # Center is found to displace image after spherical projection
    (x_c, y_c) = (widthIm / 2.0, heightIm / 2.0)  # determine centre of image

    X,Y = np.meshgrid(np.linspace(0, xdim, xdim),np.linspace(0, ydim, ydim))
    theta = (X - x_c) / f
    phi = (Y - y_c) / f

    # Spherical coordinates to Cartesian
    xcap = np.sin(theta) * np.cos(phi)
    ycap = np.sin(phi)
    zcap = np.cos(theta) * np.cos(phi)

    xn = xcap / zcap
    yn = ycap / zcap
    r = np.power(xn,2) + np.power(yn,2) # For Radial Distortion (if added)

    # Convert to floor
    ximg = np.floor(f * xn + x_c)
    yimg = np.floor(f * yn + y_c)

    # We return the restuls as float32 to have more decimals on interpolation
    return ximg.astype("float32"), yimg.astype("float32")


# We trim the eges of a spherical projection to avoid black lines in panorama/mosaic
# This is OPTIONAL, but depending on the image it may return better results
# Too much trimming would delete pixels used for matching; thus reducing them

def sphericalTrim(images, cutPix=10):
    # IF values are tu low of cero we just trim a pixel
    if cutPix[0] == 0 or cutPix[1] == 0:
        cutPix = [1, 1]

    # IF we ever have RGB images we consider it
    if len(images.shape) == 3:
        images = images[cutPix[0]:-cutPix[0], cutPix[1]:-cutPix[1], :]
    # Trim for BW images (1 channel only)
    else:
        images = images[cutPix[0]:-cutPix[0], cutPix[1]:-cutPix[1]]
    return images


# We create the spherical projections with the image, a focal lenght f and a pixel cut to avoid black lines
def sphericalProjection(images, f,cutPix=[45, 5]):
    # Here the projection-spherical images are computed by interpolating its values
    # from the source image (use cv2.remap() in each band).
    outArray = []
    for ele in images:
        # Map for x and y values is created for each image
        map_x, map_y = spherical_lookupTable(f, ele)
        # Interpolated image with the map is created with CUBIC for precision (more expensive but sharper outcome)
        imout = cv2.remap(ele, map_x, map_y, cv2.INTER_CUBIC, cv2.BORDER_TRANSPARENT)
        # Image is appended to a new array after trimming (if defined)
        outArray.append(sphericalTrim(blackOut(imout),cutPix))
    return outArray