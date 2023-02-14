"""
==========================================
            NETWORK BLOCK
==========================================
"""

import numpy as np
import operator
import networkx as nx # To aid in the recostruction of Minimum Spanning Trees (MST)

import src.networkSupport as networkSupport # Complimentary Functions for Graph-Theory Manipulation
import src.bundleAdjustment as bundleAdjustment
###########################################################################

##### LEGACY #####
# One alterative to Not use "Bundle Adjustment" is to get the appropriate order
# that maximizes the number of conections visible on top. This project used it
# in the beginning but it moved to a more general approach. These 2 functions are
# no longer in use

def getEdges(G):
    n = G.shape[0]
    edgesTmp = np.zeros((3, int(n * (n - 1) / 2)))
    c = -1
    for i in range(n):
        for j in range(i + 1, n):
            if G[i, j] > 0:
                c = c + 1
                edgesTmp[:, c] = [i, j, G[i, j]]

    edges = edgesTmp[:, 0: c + 1]
    return edges


def getOrdering(indices, tree):
    k = len(indices)
    ordering = np.zeros((k, 1))
    visited = np.zeros((k, 1))
    subtree = tree[np.ix_(indices, indices)]
    edges = getEdges(subtree)  # This is a function
    index = result = np.where(edges[2, :] == np.amax(edges[2, :]))[0]
    index_i = int(edges[0, index])
    index_j = int(edges[1, index])
    ordering[0, 0] = index_i
    ordering[1, 0] = index_j
    visited[index_i, 0] = 1
    visited[index_j, 0] = 1

    c = 2
    fringe = np.array([[], [], []])
    for index in range(k):
        if (subtree[index, index_j] > 0) & (operator.not_(bool(visited[index][0]))):
            # fringe = [fringe, [index;index_j;subtree(index, index_j)]]
            tmpStack = np.array([[index], [index_j], [subtree[index, index_j]]])
            fringe = np.hstack((fringe, tmpStack))
    while c < k:
        for index in range(k):
            if (subtree[index, index_i] > 0) & (operator.not_(bool(visited[index][0]))):
                tmpStack = np.array([[index], [index_i], [subtree[index, index_i]]])
                fringe = np.hstack((fringe, tmpStack))

        index = int(np.where(fringe[2, :] == np.amax(fringe[2, :]))[0][0])
        index_i = int(fringe[0, index])
        # Para vaciar la columna del fringe
        fringe = np.delete(fringe, index, axis=1)

        ordering[c, 0] = index_i
        visited[index_i, 0] = 1
        c = c + 1

    return ordering


###########################################################################
# A recursive function is used to get the initial H parameters in the GRAPH
# Based on a Tree (MST), there's only one possible way to follow a network
# with a recursive approach this gets the homography that connects the images
# with a defined center

####################################################################
# Recursive process to get the connections of a node with the rest
####################################################################
def getRouteMatrix(i, tree, individualHomography, visited, routeMatrix):
    # i is the index of the source and j of the destination images
    visited[i, 0] = 1
    n = tree.shape[0]

    for j in range(n):
        if (i != j) and (operator.not_(bool(visited[j, 0]))) and (tree[j, i] > 0):
            # If correct we choose the homography of the row index

            # Initialize
            tform = np.array([str(i)])
            tform = np.append(tform, routeMatrix[i, 0])
            routeMatrix[j, 0] = tform

            # Recursive Call
            # Row j becomes the column (i colum - souce)
            routeMatrix = getRouteMatrix(j, tree, individualHomography, visited, routeMatrix)

    return routeMatrix


####################################################################
# Recursive process to get the Homography of the Path Between Nodes
####################################################################


def getPathBetweenNodes(i, tree, individualHomography, visited, combinedTransfrom):
    # the new source i is the old destination j
    visited[i, 0] = 1
    n = tree.shape[0]

    for j in range(n):
        if (i != j) and (operator.not_(bool(visited[j, 0]))) and (tree[j, i] > 0):
            # Si es correcto, escogemos la homografia individual del row
            tform = individualHomography[j, i]
            tform = np.matmul(tform, combinedTransfrom[i, 0])
            combinedTransfrom[j, 0] = np.divide(tform, tform[2, 2])

            # IMPORTANTE#################
            combinedTransfrom = getPathBetweenNodes(j, tree, individualHomography, visited, combinedTransfrom)

    return combinedTransfrom


###########################################################################

####################################################################
# Allows the start of BOTH recursive functions
####################################################################
def getNodeTransform(tree, i, individualHomography):
    # We get i to be the sources we fix, columns are frozen
    n = tree.shape[0]
    # Define visited array to not go back in the tree
    visited = np.zeros((n, 1))
    visited2 = np.zeros((n, 1))
    # Array to store homography
    combinedTransfrom = np.empty((n, 1), dtype=object)
    routeMatrix = np.empty((n, 1), dtype=object)
    for m in range(int(combinedTransfrom.shape[0])):
        combinedTransfrom[m][0] = np.eye(3)
        routeMatrix[m][0] = "xxx"  # Add element to identify the ending of a route

    # with the routes calculated we can calculated the optimized Homogragy with Marquad and
    # input it into nodeHomography as individual Homography
    routeMatrix = getRouteMatrix(i, tree, individualHomography, visited2, routeMatrix)
    nodeHomography = getPathBetweenNodes(i, tree, individualHomography, visited, combinedTransfrom)

    return nodeHomography, routeMatrix


###########################################################################

####################################################################
# Minimum spanning Trees Prim's Algorithm was used to detect significant connections
# Equivalent to the k-d-tree one author proposed but helps to create subTrees
# Based on an Adjacency Matrix of numMatches obtained from Matchers
####################################################################
def getUndirectedGraph(numMatches, homographyMatches, images):
    print("Building Network...")

    # numMathces is the adjacency matrix for the MST algorithm
    # ccnum is the number of potential center images
    subTreesIdx, subTrees, concomps = networkSupport.createGraphNetwork(numMatches)

    # For each sub-tree (group of images of the same panorama)
    # We estimate the OPEN route that connects all of them (helps to avoid
    # reiterations and save computation)
    treesMST = np.empty((len(subTrees), 1), dtype=object)
    for ele in range(len(subTrees)):
        G = nx.from_numpy_matrix(subTrees[ele, 0])
        T = nx.minimum_spanning_tree(G)
        treeTmp = nx.to_numpy_array(T)
        treesMST[ele, 0] = treeTmp

    finalPanoramaTforms = np.empty((len(subTrees), 1), dtype=object)

    # Initialize Parameters
    ordering = np.empty((len(subTrees), 1), dtype=object)
    allCombinedTforms = np.empty((len(subTrees), 1), dtype=object)
    allRoutes = np.empty((len(subTrees), 1), dtype=object)
    centerIdx = np.empty((len(subTrees), 1), dtype=int)

    # We iterate for each group of images (Subtree) of each Panorama to Create
    for cc in range(len(subTrees)):

        # Homographies Corresponding to those images are retrieved
        tree = treesMST[cc, 0]
        homographyMatchesTmp = homographyMatches[np.ix_(list(subTreesIdx[cc, 0]), list(subTreesIdx[cc, 0]))]

        # We need to find the indixes of thenodes that share component
        # and that could potentially be the center of the image
        indicesTmp = np.arange(len(subTreesIdx[cc, 0]))
        k = len(indicesTmp)
        ordering[cc, 0] = []

        ####################################################
        ########   BUNDLE ADJUSTMENT
        #####################################################
        if len(subTreesIdx[cc, 0]) > 1:
            bundleHomo = bundleAdjustment.bundle_start(subTrees[cc, 0], images, subTreesIdx[cc, 0], homographyMatchesTmp,
                                      ordering[cc, 0])
        else:
            bundleHomo = homographyMatchesTmp

        finalPanoramaTforms[cc, 0] = bundleHomo

        # Initialize Parameters to obtain the routeList and combinedHomography
        # in matrix form
        finalTformsTmp = np.empty(homographyMatchesTmp.shape, dtype=object)
        allRouteMatrixTmp = np.empty(homographyMatchesTmp.shape, dtype=object)
        for index in range(k):
            # We get the index of the image that will be iterating
            i = indicesTmp[index]

            # We want to reach the point where the source column is equal to the goal
            # Each node homography tells us how the homography to go from the source to the destination
            nodeHomography, routeMatrix = getNodeTransform(tree, i, homographyMatchesTmp)
            finalTformsTmp[:, [i]] = nodeHomography  # Para vaciar la columna y poder anexas
            allRouteMatrixTmp[:, [i]] = routeMatrix

        # Based on the Number of Nodes Connected and The number of matches in total
        # the center is selected

        allCombinedTforms[cc, 0] = finalTformsTmp
        allRoutes[cc, 0] = allRouteMatrixTmp
        indexMultiple = np.argwhere(np.count_nonzero(tree, axis=1) == np.amax(np.count_nonzero(tree, axis=1)))
        colSum = 0
        index = 0
        for indice in indexMultiple:
            sumTmp = np.sum(tree[:, indice])
            if sumTmp > colSum:
                index = indice
                colSum = sumTmp

        center = indicesTmp[index]

        centerIdx[cc, 0] = center
        print("Center index is: ", center)

    print("Building Network: DONE")
    return concomps, finalPanoramaTforms, centerIdx, allCombinedTforms, subTreesIdx, ordering, allRoutes
