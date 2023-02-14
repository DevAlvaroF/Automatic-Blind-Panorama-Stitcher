import numpy as np
import os
import src.P0 as P0
import random
from os import listdir
from os.path import isfile, join
import cv2

from src import config


def readBWColor(folderName,randomizar=False):
    print("Reading Images and Resizing....")
    cwd = folderName

    nameImages = [f for f in listdir(cwd) if isfile(join(cwd, f))]
    if randomizar:
        random.shuffle(nameImages)  # randomizes list in its place

    images = []
    imagesColor = []
    get_image = lambda route: os.path.join(cwd, route)

    for item in nameImages:
        tmp = np.uint8(P0.readIm(get_image(item), 0))
        tmpColor = np.uint8(P0.readIm(get_image(item), 1))
        if config.resizeImage == 1:
            basewidth = config.basewidth
            wpercent = (basewidth/float(tmp.shape[0]))
            hsize = int((float(tmp.shape[1]) * float(wpercent)))

            tmp = cv2.resize(tmp, (hsize,basewidth),interpolation = cv2.INTER_CUBIC)
            tmpColor = cv2.resize(tmpColor, (hsize,basewidth), interpolation=cv2.INTER_CUBIC)

            #Para guardar archivo
            #ruta = os.path.join("C:/Users/alvar/iCloudDrive/COSI/2nd Semester UGR/Computer Vision/Final Project/Own Development/images/CustomPano2/small/", str(item))
            #cv2.imwrite(ruta, cv2.cvtColor(tmpColor, cv2.COLOR_BGR2RGB))

        images.append(tmp)
        imagesColor.append(tmpColor)  # we convert to int because opencv argues later on to display in color

    print("Reading Images and Resizing: DONE")
    return images, imagesColor
