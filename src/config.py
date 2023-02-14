# Warping Parameters
warpType = 'planar'   # 'spherical' | 'planar'
f = 600 # DONT TOUCH

# Resize if images too small or big to baseWidth (Keeps aspec ratio)
resizeImage = 0  #DONT TOUCH
basewidth = 640

RANSAC_error = 1.0 #max allowed error (1 is good)

#Gain Compensation
GAIN_sigmaN = 12
GAIN_sigmaG = 0.1

# Multiband Blending
BLEND_sigma = 0.2 # Optimal Value: 0.2 (depends size of image)
BLEND_bands = 7 # Optimal Value: 7 (3minimum)
BLEND_kSize = 0 # Acceptable Value: 0 (Computed from Sigma)

# Homography Optimization with Levenberg-Marquad
transformationRefinement = 0

# Transformation Selection
transform = 'homography' 

# Matcher
MATCHER = 'knn'  # This can be 'knn' or 'flann"
flannTREES = 10
KNN_ratio = 0.7 # accordingt to https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
FLANN_ratio = 0.7


# Additional Samples
EXTRA_planarBoundingBox = True # (Requires 'planar' view)
