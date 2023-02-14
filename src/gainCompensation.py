import numpy as np
import src.config as config
import src.utilities as utilities
import cv2
from joblib import parallel_backend


def globalResults(a,b,c,i):
    global Ibarijvalue, Ibarjivalue, Nijvalue
    Ibarijvalue[i, 0] = a
    Ibarjivalue[i, 0] = b
    Nijvalue[i, 0] = c

def parForFunction(matSize,IuppeIdx,panoramasize,warpedImages,Ibarijvalue,Ibarjivalue,Nijvalue):
    for i in range(len(IuppeIdx)):

    # Rows, columns index is extracted from the function
        # ii is rows
        # jj is columns
        ii,jj = utilities.ind2sub(matSize,np.asarray(IuppeIdx[i]))
        if ii != jj:
            Ibarij = np.zeros(panoramasize, dtype=np.uint8)
            Ibarji = np.zeros(panoramasize, dtype=np.uint8)

            # Binarize using Otus method
            tmpi = cv2.cvtColor(warpedImages[ii,0].copy(), cv2.COLOR_RGB2GRAY)
            tmpj = cv2.cvtColor(warpedImages[jj,0].copy(), cv2.COLOR_RGB2GRAY)

            ############ Threshold Calculation
            threshBin = 0
            reti, maski = cv2.threshold(tmpi, threshBin, 255, cv2.THRESH_BINARY)  # turn 60, 120 for the best OCR results
            kernel = np.ones((5, 3), np.uint8)
            #maski = cv2.erode(maski, kernel, iterations=1)

            retj, maskj = cv2.threshold(tmpj, threshBin, 255, cv2.THRESH_BINARY)  # turn 60, 120 for the best OCR results
            #maskj = cv2.erode(maskj, kernel, iterations=1)

            Nij_im = maski & maskj


            Nij_im = np.asarray(Nij_im,dtype=bool)

            Imij = warpedImages[ii,0]
            Imji = warpedImages[jj,0]

            # Use the mask to extrac values of the warped images
            # assign the image values to Ibar with same bool index

            #### Filtering Channels
            for cIdx in range(Imij.shape[2]):
                Ibarij[:,:,cIdx][Nij_im] = Imij[:,:,cIdx][Nij_im]
                Ibarji[:,:,cIdx][Nij_im] = Imji[:,:,cIdx][Nij_im]


            Ibarij_double = np.asarray(Ibarij,dtype=float)
            Ibarji_double = np.asarray(Ibarji,dtype=float)

            Nij_sum = sum(sum(Nij_im))

            Ibarij_double = np.array([[[np.sum(Ibarij_double[:,:,0])]], [[np.sum(Ibarij_double[:,:,1])]],[[np.sum(Ibarij_double[:,:,2])]]])
            Ibarji_double = np.array([[[np.sum(Ibarji_double[:,:,0])]], [[np.sum(Ibarji_double[:,:,1])]],[[np.sum(Ibarji_double[:,:,2])]]])

            # We add error handling because sometimes there is an error with division by 0
            # try and except doesn't work because it is a warning, not an error
            with np.errstate(invalid='ignore'):

                Ibarijvalue[i,0] = np.reshape(Ibarij_double / Nij_sum, (1, 3))
                Ibarjivalue[i, 0] = np.reshape(Ibarji_double / Nij_sum, (1, 3))

            Nijvalue[i,0] = Nij_sum

    return Ibarijvalue,Ibarjivalue,Nijvalue

def init_pool(a,b,c):
    global Ibarijvalue, Ibarjivalue, Nijvalue
    Ibarijvalue = a
    Ibarjivalue = b
    Nijvalue = c


def gainCompensation(warpedImages):
    print("===Starting Gain Compensation===")
    # Initialize Parameters
    n = len(warpedImages)

    gainImages = np.empty((1,n),dtype=object)
    sigmaN = config.GAIN_sigmaN
    sigmag = config.GAIN_sigmaG

    # We get Panorama Size from the first transfromed image
    panoramasize = warpedImages[0,0].shape

    Iij = np.empty((n,n),dtype=object)
    Iji = np.empty((n,n),dtype=object)
    Nij = np.zeros((n,n),dtype=float)

    matSize = Iij.shape
    tmpIupperIdx = np.triu(np.reshape(list(range(1,(n*n)+1)), (n,n),order='F'))
    IuppeIdx = list(np.sort(tmpIupperIdx[np.nonzero(tmpIupperIdx)]) - 1)
    #IuppeIdx = nonzeros(triu(reshape(1:numel(Iij), size(Iij))));

    # This will be initialized with memory locationsn for the pool processing
    a = np.empty((len(IuppeIdx),1),dtype=object)
    b = np.empty((len(IuppeIdx),1),dtype=object)
    c = np.zeros((len(IuppeIdx),1))

    Ibarijvalue = a
    Ibarjivalue = b
    Nijvalue = c

##################################
    ################################### PARALLEL PROCESSING
    with parallel_backend('multiprocessing'):
        Ibarijvalue, Ibarjivalue, Nijvalue = parForFunction(matSize,IuppeIdx,panoramasize,warpedImages,Ibarijvalue,Ibarjivalue,Nijvalue)


    # Populate the upper triangle matrix
    Iij = utilities.fillUpTri(Iij,Ibarijvalue)
    Iji = utilities.fillUpTri(Iji, Ibarjivalue)
    Nij = utilities.fillUpTri(Nij, Nijvalue)

    Iijc = Iij
    Ijic = Iji

    iii = np.ones(Iij.shape)
    idx = np.where(np.tril(iii, -1))

    Iijp = np.transpose(Iij)
    Ijip = np.transpose(Iji)

    Iijc[idx] = Iijp[idx]
    Ijic[idx] = Ijip[idx]

    Iij = Iijc
    Iji = Ijic

    Nij = Nij + np.tril(np.transpose(Nij),-1)

    print("Removing InProcess Clutter...")
    # we change the nan for zeros
    Iij = utilities.matVecNaN2Zero(Iij)
    Iji = utilities.matVecNaN2Zero(Iji)

    gainmatR = np.zeros((n,n))
    gainmatG = np.zeros((n,n))
    gainmatB = np.zeros((n,n))

    gainR = np.zeros((len(IuppeIdx), 1))
    gainG = np.zeros((len(IuppeIdx), 1))
    gainB = np.zeros((len(IuppeIdx), 1))

    for i in range(len(IuppeIdx)):
        ii, jj = utilities.ind2sub(matSize, np.asarray(IuppeIdx[i]))
        if ii != jj:
            gainR[i, 0] = -(Nij[ii, jj] * Iij[ii, jj][0, 0] * Iji[jj, ii][0, 0] + Nij[jj, ii] * Iij[jj, ii][0, 0] * Iji[jj, ii][0, 0]) / np.power(sigmaN, 2)
            gainG[i, 0] = -(Nij[ii, jj] * Iij[ii, jj][0, 1] * Iji[jj, ii][0, 1] + Nij[jj, ii] * Iij[jj, ii][0, 1] * Iji[jj, ii][0, 1]) / np.power(sigmaN, 2)
            gainB[i, 0] = -(Nij[ii, jj] * Iij[ii, jj][0, 2] * Iji[jj, ii][0, 2] + Nij[jj, ii] * Iij[jj, ii][0, 2] * Iji[jj, ii][0, 2]) / np.power(sigmaN, 2)
        else:
            gainRval = 0
            gainGval = 0
            gainBval = 0

            for iii in range(n):
                if iii != ii:
                    gainRval = gainRval + (((Nij[ii, iii] * np.power(Iij[ii, iii][0, 0],2) + Nij[iii, ii] * np.power(Iji[ii, iii][0, 0],2)) / np.power(sigmaN, 2)) + (Nij[ii, iii] / np.power(sigmag, 2)))
                    gainGval = gainGval + (((Nij[ii, iii] * np.power(Iij[ii, iii][0, 1], 2) + Nij[iii, ii] * np.power(Iji[ii, iii][0, 1], 2)) / np.power(sigmaN, 2)) + (Nij[ii, iii] / np.power(sigmag, 2)))
                    gainBval = gainBval + (((Nij[ii, iii] * np.power(Iij[ii, iii][0, 2], 2) + Nij[iii, ii] * np.power(Iji[ii, iii][0, 2], 2)) / np.power(sigmaN, 2)) + (Nij[ii, iii] / np.power(sigmag, 2)))

            gainR[i,0] = gainRval
            gainG[i,0] = gainGval
            gainB[i,0] = gainBval

    # Populate the upper triangle matrix
    gainmatR = utilities.fillUpTri(gainmatR, gainR)
    gainmatG = utilities.fillUpTri(gainmatG, gainG)
    gainmatB = utilities.fillUpTri(gainmatB, gainB)

    gainmatR = gainmatR + np.tril(np.transpose(gainmatR),-1)
    gainmatG = gainmatG + np.tril(np.transpose(gainmatG), -1)
    gainmatB = gainmatB + np.tril(np.transpose(gainmatB), -1)

    #------------------------------------------------------------
    #  AX = b --> X = A \ b
    b = np.zeros((n, 1))
    for j in range(n):
        for i in range(n):
            b[j,0] = b[j,0] + (Nij[j, i] / np.power(sigmag,2))

    gR = np.linalg.lstsq(gainmatR, b,rcond=None)[0]
    gG = np.linalg.lstsq(gainmatG, b,rcond=None)[0]
    gB = np.linalg.lstsq(gainmatB, b,rcond=None)[0]


    #--------------------------------------------------------------
    #  Compensate gains for images
    for i in range(n):
        # multiplicamos cada canal por su gain
        tmpWrpImg = np.asarray(warpedImages[i,0],dtype=float)
        tmpWrpImg[:,:,0] = tmpWrpImg[:,:,0]*gR[i,0]
        tmpWrpImg[:, :, 1] = tmpWrpImg[:,:,1]*gG[i, 0]
        tmpWrpImg[:, :, 2] = tmpWrpImg[:,:,2]*gB[i, 0]
        gainImages[0,i] = np.asarray(tmpWrpImg,dtype=np.uint8)
        #P0.displayIm(gainImages[0,i],title="Imagen compensation")


    return gainImages
