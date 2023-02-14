"""
==========================================
         BUNDLE ADJUSTMENT BLOCK
==========================================
"""

import numpy as np


###########################################################################

# Homographies are built for "Bundle Adjustment" based on the connections
def build_homographies(connected_components, pair_matches, individualHomographies):

    # Inintialization
    pair_matches = [ele[0:2].astype(np.uint8) for ele in pair_matches]
    component_matches = pair_matches.copy()
    connected_component = connected_components.copy()

    images_added = []
    current_homography = np.eye(3)

    # Based on the Match of an image (Graph Node over an Edge)
    # The Homography is obtained (previously calculated)
    pair_match = component_matches[0]
    pair_match_H = individualHomographies[int(pair_match[0]),int(pair_match[1])]


    # Initialize Images and Homographies Matrices
    imagesHomo = np.empty((len(connected_components),len(connected_components)),dtype=object)
    imagesHomo = [np.eye(3) for ele in imagesHomo]
    nb_pairs = len(pair_matches)

    # Calculate the number of Pairs for each Element in a Pair of Images
    # Count the occurances and by a factor, get which one has the most matches
    # The one with the highest number keeps Identity matrix

    pair0_count = sum([10 * (nb_pairs - i) for i, match in enumerate(pair_matches) if (pair_match[0] in match)])
    pair1_count = sum([10 * (nb_pairs - i) for i, match in enumerate(pair_matches) if (pair_match[1] in match)])

    if pair0_count > pair1_count:
        imagesHomo[pair_match[0]] = np.eye(3) # image_a
        imagesHomo[pair_match[1]] = pair_match_H # image_b

    else:
        imagesHomo[pair_match[1]] = np.eye(3)  # image_b
        imagesHomo[pair_match[0]] = np.linalg.inv(pair_match_H)  # image_a

    images_added.append(pair_match[0])  # image_a
    images_added.append(pair_match[1]) #image_b

    # Iterate until we've done the "connected Homography" multiplication for each
    # pair. If the image is after or before, the homography could be inversed
    while len(images_added) < len(connected_component):
        for pair_match in component_matches:

            imagea = pair_match[0]
            imageb = pair_match[1]

            if (imagea in images_added) and not (imageb in images_added):

                pair_match_H = individualHomographies[int(pair_match[0]), int(pair_match[1])]
                homography = pair_match_H @ current_homography #matrix multiplication
                imagesHomo[pair_match[1]] = imagesHomo[pair_match[0]] @ homography
                images_added.append(imageb)
                break

            elif not (imagea in images_added) and (imageb in images_added):
                pair_match_H = individualHomographies[int(pair_match[0]), int(pair_match[1])]
                homography = np.linalg.inv(pair_match_H) @ current_homography  # matrix multiplication
                imagesHomo[pair_match[0]] = imagesHomo[pair_match[1]] @ homography
                images_added.append(imagea)
                break

    return imagesHomo

###########################################################################

# The number of matches that correspond to each individual source without repeating
# We only use the lower triangle of the matrix to ensure single matches and faste
def find_significant_matches(numMatches, ordering, subTreesIdx,maxMatches = 5):
    pair_matches = []
    workingMatrix = np.tril(numMatches)
    k = numMatches.shape[1]
    for i in range(k):
        for m in range(k):
            if (workingMatrix[m,i] >0) and (i != m):
                tmp = np.array([m,i, workingMatrix[m,i]]) # first is source (Column)
                pair_matches.append(tmp)

    # Order based on number of Matches
    pair_matches.sort(key=lambda x: x[2], reverse=True)
    return pair_matches

###########################################################################

# This function puts together the Homography Calculation for the bundle of
# images of each subTree
def bundle_start(numMatches, images,subTreesIdx,individualHomographies, ordering):

    individualImages = []
    for ii in subTreesIdx:
        individualImages.append(images[ii])

    print("Finding Significant Matches...")
    pair_matches = find_significant_matches(numMatches, ordering, subTreesIdx)
    print("Bundle Adjustment...")
    bundleHomo = build_homographies(individualImages, pair_matches,individualHomographies)

    # Casting into a numpy array
    bundleHomography = np.empty((1,len(bundleHomo)),dtype=object)
    for i in range(len(bundleHomo)):
        bundleHomography[0,i] = bundleHomo[i]
    return bundleHomography