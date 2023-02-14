# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 18:47:10 2022

@author: alvar
"""

import cv2, numpy as np
from matplotlib import pyplot as plt

def readIm(filename, flagColor=1):
  # cv2 reads BGR format
  im=cv2.imread(filename)
  #print(filename)
  if flagColor == 0:
    #Returns image converted to gray space
    tmp = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    return tmp.astype("float64")
  elif flagColor == 1:
    #returns same image (no transformation needed)
    tmp =  cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    return tmp.astype("float64")

#Todo será manejado entreo 0-1, y solo en el plot se multiplicaría por 255 en
#caso de ser necesario
def rangeDisplay01(image, flag_GLOBAL):
  image = ((image) - np.amin(image))/(np.amax(image)-np.amin(image))
  return image

def displayIm(im, title='Result',showFlag=True,cmap="gray",normalizar=False,factor=1):
  # First normalize range
  vmaxValue = 255
  if(normalizar):
      im = rangeDisplay01(im,0)
      vmaxValue = 1

  if(False): #Para ver los valores altos y bajos y confirmar normalizacion
    print(np.amin(im))
    print(np.amax(im))
    
  #plt.figure(dpi=100)
  #Added per requirement of Teacher for display size
  figure_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(factor * figure_size)
  plt.title(title)
  plt.xticks([]), plt.yticks([]) #  axis label off
  if showFlag:
    #plt.imshow(im,cmap, vmin = 0, vmax = 255)
    plt.imshow(im,cmap="gray",vmin = 0, vmax = vmaxValue)
    plt.show()

def displayMI_ES(vim,title,normalizar=False,factor=1):
  #Normalizamos las imagenes antes de visualizar. Entreo 0 y 1 (imshow lo hace entonces mejor damos el input)
  if(normalizar):
    vim = [rangeDisplay01(ele,0) for ele in vim]
  
    
  # The simplest case, one row
  if len(vim) == 4:
    imagen_concat = np.hstack((vim[0],vim[1],vim[2],vim[3]))
    displayIm(imagen_concat,'Result')


  else:
    # Now compute the number of block-rows
    rows_full = 0
    lonely = 0;
    if np.mod(len(vim),4) == 0: #exactly 4 per row
      rows_full = int(np.floor(len(vim)/4))
      lonely = 0
    else: #last row will not be completely full
      rows_full = int(np.floor(len(vim)/4))
      lonely = len(vim) - (rows_full*4)


    # we build up the first block-row
    tmp_matrix=vim[0];
    out = [];
    bandera = 1;

    #Iteramos por cada fila completa
    for item in range((rows_full*4)):

      if bandera == 4: #creo hay que bajarlo a 3
        if len(out) == 0:
          out = tmp_matrix
        else:
          out = np.vstack((out,tmp_matrix))
          
        bandera = 0
        try:
          tmp_matrix=[];
        except:
          print('aqui cosita')
      
      if len(tmp_matrix) == 0:
          tmp_matrix=vim[item+1];
      else:
          tmp_matrix = np.hstack((tmp_matrix,vim[item+1]))
      bandera = bandera + 1
        
    # construimos última fila incompleta
    matriz_incompleta = vim[rows_full*4]


    for incom in range(lonely-1):
      indice_lonely = (rows_full * 4) + incom
      if incom == 0:
        matriz_incompleta = vim[indice_lonely]
      else:
        matriz_incompleta = np.hstack((matriz_incompleta,vim[indice_lonely]))
    #Normalizamos
    #if(True):
      #out = rangeDisplay01(out,0)
      #matriz_incompleta = rangeDisplay01(matriz_incompleta,0)
    

    
    #añadimos una matriz de 255 para completar la fila
    rows_imagen = vim[0].shape[0]
    column_imagen = vim[0].shape[1]

    if len(matriz_incompleta.shape) == 3:
      unos_completas = (np.zeros((rows_imagen,column_imagen*(4-lonely),3),np.int64)+0.0).astype("float64")
    else:
      unos_completas = (np.zeros((rows_imagen,column_imagen*(4-lonely)),np.int64)+0.0).astype("float64")
    
    incompleta_full = np.hstack((matriz_incompleta,unos_completas))

    final_matriz = np.vstack((out,incompleta_full))

    displayIm(final_matriz,title,normalizar=normalizar,factor=factor)


def displayMI_NES(vim,title='SinTitulo',normalizar = False,factor=1):
  #Normalizamos las imagenes antes de visualizar. Entreo 0 y 1 (imshow lo hace entonces mejor damos el input)
  if(normalizar):
    vim = [rangeDisplay01(ele,0) for ele in vim]
    
  #We convert each image in 3 channels for this code to work with everything (Same value copied in 3D)
  contador = 0
  for item in vim:
    if len(item.shape) == 2:
      vim[contador] = np.repeat(item[:, :, np.newaxis], 3, axis=2)
      contador = contador + 1
      #displayIm(item.astype(np.uint8))

  vimSize = []
  #Estimate the images size in column number
  for item in range(len(vim)):
    size = vim[item].shape[1]
    vimSize.append(size)


  #Order array from biggest image to smallest based on columns FROM vimSize
 
  currentIndex = 0;
  iterationsPossible = np.math.factorial(len(vimSize))
  for i in range(iterationsPossible):
    #We run it aslong as we are not in the last position
    if currentIndex != len(vimSize)-1:
      if vimSize[currentIndex] < vimSize[currentIndex + 1]:
        #we change positions to have the highest first
        vimSize[currentIndex],vimSize[currentIndex+1] = vimSize[currentIndex+1],vimSize[currentIndex]
        vim[currentIndex],vim[currentIndex+1] = vim[currentIndex+1],vim[currentIndex]
      currentIndex = currentIndex + 1
    else:
        currentIndex = 0


  #Estimate number of "subimages" of 2 elements
  modSubIm = np.mod(len(vim),2)
  if modSubIm == 0:
    numSubIm = len(vim)/2
  else:
    #numSubIm = 1 + len(vim)/2
    numSubIm = np.ceil(1 + len(vim)/2)

  #Create array of sub images
  #subIm
  #Creamos la primer columna que no es la principal
  colDiff = vim[0].shape[1]-vim[1].shape[1]
  #tmp = np.pad(vim[1], ((0, 0),(0,colDiff),(0,0)), mode='constant',constant_values=0.0)
  tmp = vim[1]

  contador = 2
  for subIm in range(int(numSubIm)-1):

    im1 = vim[contador]
    colDiff = tmp.shape[1]-im1.shape[1]
    #we add padding on the rows at the end, the rest of the dimensions nothing
    im1 = np.pad(im1, ((0, 0),(0,colDiff),(0,0)), mode='constant',constant_values=0.0)
    tmp = np.vstack((tmp,im1))
    contador = contador + 1
    
  if (tmp.shape[0]<vim[0].shape[0]):
    print("Last Image")
    im1 = np.zeros((vim[0].shape[0]-tmp.shape[0],tmp.shape[1],3), np.float64)
    tmp = np.vstack((tmp,im1))

  finalAppend = np.hstack((vim[0],tmp))

  displayIm(finalAppend,title=title,normalizar=normalizar,factor=factor)