"""
==========================================
  FEATURE PROCESSING AND MATCHING BLOCK
==========================================
"""
global imagen1 #global variables for Parallel Processing
global imagen2

import src.config as config
from itertools import compress
import src.utilities as utilities
from matplotlib import pyplot as plt
from random import randint
from src.optimize_fcn import * # For the optional non-linear optimization

###########################################################################

# Extract Features of a list of Images
def siftFeatureExtraction(images):
    print("Extracting Features...")
    numImg = len(images)
    modifiedImages = []
    kpComplete = []
    dsComplete = []

    # Iterate through each Image File
    for idx in range(numImg):
        # We create a sift object detector, parameters are left as standard
        sift = cv2.SIFT_create()
        # Extract Keypoints and Descriptors
        # Image is converted to uint8 as requested by opencv
        imgTmp = np.uint8(images[idx])
        # With the sift object we'll detect keypoints and extract the features (Descriptores)
        # of the selected image
        kpTmp, dsTmp = sift.detectAndCompute(imgTmp, None)

        kpComplete.append(kpTmp)
        dsComplete.append(dsTmp)

    print("Extracting Features: DONE")
    return kpComplete, dsComplete


###########################################################################

# Match 2 Images based on their Descriptors
def matching(ds1, ds2):

    # Two method are included: 'knn' and 'flann' for bruteforce and nearest
    # neighbors approach
    testPass = []
    if (config.MATCHER == 'knn'):
        # Create Matcher object and Obtain Matches
        matcher = cv2.BFMatcher.create(normType=cv2.NORM_L2,crossCheck=False)
        matches = matcher.knnMatch(ds1, ds2,k=2)

        # Apply ratio test as suggested by Lowe's to eliminate wrong matches
        testPass = []
        for m, n in matches:  # We expect 2 parameters because k=2
            # If the distance of two features is compared, and we only keep if it passes the criteria
            if (m.distance < (config.KNN_ratio * n.distance)):
                testPass.append(m)

    # Depending on the scene conditions the "flann" matcher could lead to better
    # results for spherical projections
    elif (config.MATCHER == 'flann'):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # or pass empty dictionary

        # Create Objects and get Matches
        flannMatcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flannMatcher.knnMatch(ds1, ds2, k=2)

        # Perform Ratio Test to reduce wrong matches
        ratio_thresh = config.FLANN_ratio
        testPass = []
        for m, n in matches:
            if m.distance < (ratio_thresh * n.distance):
                testPass.append(m)
    return testPass

###########################################################################

# To get Transformation Matrix between 2 images using keypoints
def getTransformMatrix(kp1, kp2, match):
    # Initialize Variables
    src_pts = []
    dst_pts = []
    correspondence = np.empty((len(match),4),dtype=float)

    # Iterate through all the matches and extract the inter-pixel location
    for i,m in enumerate(match):
        srcTmp = kp1[m.queryIdx].pt
        src_pts.append(srcTmp)
        dstTmp =kp2[m.trainIdx].pt
        dst_pts.append(dstTmp)
        correspondence[i,:] = [*srcTmp, *dstTmp] # Create an [X_src Y_src X_dst Y_dst] vector

    # Convert to Float32 for accuracy
    src_pts, dst_pts = np.float32((src_pts, dst_pts))

    # Initialize homography
    H = np.eye(3)
    # Obtain the Homography from one point to another with RANSAC to calculate inliers
    if (config.transform == 'affine'):
        tmpH, mask = cv2.estimateAffine2D(src_pts,dst_pts,method = cv2.RANSAC,ransacReprojThreshold=config.RANSAC_error)
        # Python returns 2x3, we convert to 3x3
        H = utilities.toSquareTrans(tmpH)
        inliers = mask
    elif (config.transform == 'homography'): # Due to different images, Homography is the Recommended Transformation
        # H already is 3x3
        H, mask = cv2.findHomography(src_pts,dst_pts,method = cv2.RANSAC,ransacReprojThreshold=config.RANSAC_error)

    # .......... Optional Reoptimization ..........

    # If selected (not recommended as the improvement is very small), a NON-LINEAR
    # optimization between the inliers and the Homography will start
    if (config.transformationRefinement == 1) and (~(H==np.eye(3)).all()) and (hasattr(H, 'shape')):

        # Inlier Matrix is Built
        inliers = np.empty((np.sum(mask), 4), dtype=float)
        inliers[:, 0:2] = src_pts[np.where(mask)[0],:]
        inliers[:, 2:4] = dst_pts[np.where(mask)[0],:]

        # Based on the number of inliers (mask) we select random inlier points
        maskSize = np.sum(mask) // 2
        if maskSize > 6:
            nRand = randint(6, np.sum(mask) // 2)
        else:
            nRand = maskSize
        r = [randint(0, np.sum(mask)-1) for p in range(0, nRand)]  # random rows to select

        # Matrix with Random points is built
        sample_pts = np.empty((nRand, 4), dtype=float)
        sample_pts[:, 0] = src_pts[r, 0]
        sample_pts[:, 1] = src_pts[r, 1]
        sample_pts[:, 2] = dst_pts[r, 0]
        sample_pts[:, 3] = dst_pts[r, 1]

        # Optimize the homography using Levenberg-Marquardt optimization
        try:
            # Optimization with Noise Leads to a Better Value due to the descent gradient
            #x = np.concatenate((inliers, sample_pts), axis=0)

            # Pure inlier strategy is another option
            x = inliers
            opt_obj = OptimizeFunction(fun=fun_LM_homography, x0=H.flatten(), jac=jac_LM_homography,args=(x[:, 0:2], x[:, 2:]))
            LM_sol = opt_obj.levenberg_marquardt(delta_thresh=1e-24, tau=0.8)

            H_Final = None
            H_Final = H_Final / H_Final[-1, -1]
            H = H_Final

        except:
            print("Exception on LM Calculation")

    return mask, H, correspondence  #mask is a list of inliers with 0 and 1

###########################################################################

# Many Descriptors start to look similar with multiple images from differente scenes
# To improve the Selection, the Number of Inliers in the overlap area between images
# can be used to keep keypoints which are close in the feature-vector and Space
def check_inliers_in_overlap(H,imgSrc,imgDst,correspondence, mask, dsSrc, dsDest,kpSrc, kpDest):

    # The minimum number of points to build an homography is checked (4 pairs)
    if True:
        tmpMatch = matching(dsDest, dsSrc)
        if (len(tmpMatch)>= 4):
            mask, H, correspondence = getTransformMatrix(kpDest, kpSrc, tmpMatch)
    # Only if it has shape attribute (not a None or blank [])
    if (hasattr(H, 'shape')) and (len(tmpMatch) >= 4):
        alpha = 8.0
        beta = 0.3

        # Calculate overlap from Image B to A
        image_a = imgSrc
        image_b = imgDst

        # Masks are built and Transformed
        mask_a = np.ones_like(image_a[:, :], dtype=np.uint8)
        mask_b = mask_a.copy()
        mask_b = cv2.warpPerspective(np.ones_like(image_b[:, :], dtype=np.uint8), dst=mask_b, M=H, dsize=mask_a.shape[::-1])

        # Overlap Mask is obtained as well as the pixel-area (Not used but could be another segregation criteria)
        overlap = mask_a * mask_b
        area_overlap = overlap.sum()

        # Debugging Function [IGNORE]
        if False:
            P0.displayIm(showMatches(imgDst, kpDest, imgSrc, kpSrc, list(compress(tmpMatch, mask)), N=500,
                                     title="NO Outliers - Image "), factor=5)
            plt.imshow(mask_a)
            plt.show()
            plt.imshow(mask_b)
            plt.show()


        status = mask
        # recordando que A es destino y B es source
        matchpoints_a = correspondence[:, 2:]

        # The Mathces in theo verlap zone are calculated using the matchpoint indices and checking if they are part
        # of the mask where values equal 1
        matches_in_overlap = matchpoints_a[overlap[matchpoints_a[:, 1].astype(np.int64),matchpoints_a[:, 0].astype(np.int64),]== 1]

        # We apply Brown's Criteria to determine if it is a correct of wrong match
        valido = status.sum() > (alpha + (beta * matches_in_overlap.shape[0]))

    # If Homography not calculated or empty, it's not a match
    # This is due to lack of matching features
    else:
        valido = False

    return valido

###########################################################################

# Function that puts together the Detection, Extraction and Matching of
# keypoints and Features/Descriptors
def featureMatching(images, kpComplete, dsComplete):
    print("Matching Features...")

    # Initilize Variables
    numImg = int(len(images))
    kpInlier = []
    dsInlier = []

    # Output Array
    # Create NxN matrix for the matches (easier to compute interactions this way)
    numMatches = np.zeros((numImg, numImg))
    allMatches = np.empty((numImg, numImg), dtype=object)

    # Create Homography Matrix (not all cells have values, only matches)
    # Fill homography array with IDentity 3x3
    homographyMatches = np.empty((numImg, numImg), dtype=object)
    for i in range(numImg):
        for j in range(numImg):
            homographyMatches[i, j] = np.eye(3)

    minFeatures = 4  # It is defined because we want an homography

    # Custom ind2sub is used to avoid big nested loops (Faster execution)
    for xx in range(numImg*numImg):
        j, i = utilities.ind2sub((numImg, numImg), xx)

        # To ensure we don't compare the descriptors of an image with itself
        # This happens in the diagonal (less useless calculations)
        if (i != j):
            # Obtain Descriptors found in each image
            dsSrc = dsComplete[i]
            dsDest = dsComplete[j]
            # Estimate the Matches between 2 descriptors of 2 images Src->Dest
            tmpMatch = matching(dsSrc, dsDest)

            kpSrc = kpComplete[i]
            kpDest = kpComplete[j]

            # If enough points for an Homography we proceed
            if (len(tmpMatch) >= minFeatures) and (len(kpSrc)>= minFeatures)and (len(kpDest)>= minFeatures):

                # Estimate Homography and Inliers Indices
                inliers, H,correspondence = getTransformMatrix(kpSrc, kpDest, tmpMatch)

                ##########################################
                # OVERLAP AREA VERIFICATION OF INLIERS
                ##########################################

                imgSrc = images[i]
                imgDst = images[j]

                checking = check_inliers_in_overlap(H,imgSrc,imgDst,correspondence, inliers,dsSrc, dsDest,kpSrc, kpDest)

                # If Brown's condition is met for inliers in overlap area we append values
                if checking:
                    # Append matches between images with inliers index
                    allMatches[j, i] = list(compress(tmpMatch, inliers))
                    # Number of Matches That are Inliers
                    numMatches[j, i] = len(inliers)
                    # Save Homography
                    homographyMatches[j, i] = H
                else:
                    allMatches[j, i] = None

    print("Matching Features: DONE")
    return allMatches, numMatches, homographyMatches