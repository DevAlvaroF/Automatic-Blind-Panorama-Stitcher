import sys,os
# Let's import the python module exercises.py  to have access to its functions and Classes
#path_to_module='/content/drive/MyDrive/MasterCOSI/'
#sys.path.append(os.path.abspath(path_to_module))
import P0

# We import other modules to use
import cv2, numpy as np, math
from matplotlib import pyplot as plt

def gaussianMask1D(sigma=0, sizeMask=0, order=0):
  #Discretizing Criteria from -3sigma to +3 sigma with the given sigma or shape
  a = 3
  
  #If given both Sigma and SizeMask
  #e.g. If sigma= 2.2 and sizeMask=5
  #Outcome: [-6.6 -3.3  0.   3.3  6.6]
  if sizeMask != 0 and sigma != 0:
    step = a*sigma*2/(sizeMask-1)
    sigmaRange = np.arange(-a*sigma,(a*sigma)+step,step)
   
  #If only given Sigma
  #e.g. If sigma= 2.4 and sizeMask=0 (sigma gets rounded nearest integer)
  #and steps of 1 are taken
  #Outcome: [-6. -5. -4. -3. -2. -1.  0.  1.  2.  3.  4.  5.  6.]
  elif sizeMask == 0 and sigma != 0:
    sigma = np.round(sigma)  
    sigmaRange = np.arange(-a*sigma,(a*sigma)+1,1)
    
  #If only given sizeMask
  #e.g. If sigma= 0 and sizeMask=7 (sigma=1 is assumed )
  #Outcome: [-3. -2. -1.  0.  1.  2.  3.]
  elif sizeMask != 0 and sigma == 0:
    sigma = 1
    step = a*sigma*2/(sizeMask-1)
    sigmaRange = np.arange(-a*sigma,(a*sigma)+step,step)
 
  #For order
  
  #Vector of evaluation uses the sigmaRange calculated previously with the
  #considerations
  mask = []
  evaluation = np.exp(-0.5*(np.divide(sigmaRange,sigma)**2)) 

  ###################################
  #Evaluation is done for: Zero ORder
  ###################################
  if order == 0:
    mask = evaluation
    mask = mask * (1/np.sum(evaluation)) # to normalize gaussian

  ###################################
  #Evaluation of 1st Gaussian Derivative
  ###################################
  elif order == 1:
    mask = (-sigmaRange / (sigma)) * evaluation #Derivative without effect of sigma
    mask = np.flip(mask) #Due to the signs in the convolution math equation
    mask = mask * (1/np.sum(evaluation)) # to normalize gaussian

  ###################################
  #Evaluation of 2nd Gaussian Derivative
  ###################################
  elif order == 2:
    mask = (np.divide((np.power(sigmaRange,2)),(np.power(sigma,2)))   - 1) * evaluation #Derivative without effect of sigma
    mask = np.flip(mask) #Due to the signs in the convolution math equation
    mask = mask * (1/np.sum(evaluation)) # to normalize gaussian

  # return the mask as a numpy array for efficiency and manipulation
  mask = np.array([mask])
  mask = mask.reshape((mask.shape[1],1))
  return sigmaRange,mask

def plotGraph(graph, title='No title',factor=1):
  plt.figure
  figure_size = plt.gcf().get_size_inches()
  plt.gcf().set_size_inches(factor * figure_size)
  plt.plot(graph[0],graph[1])
  plt.title(title)
  plt.show()

def my2DConv(im, sigma, orders):
  #We create the output mask the same as the image, 64 bit floating point
  outim = np.zeros(im.shape,np.float64)

  #######################################
  #If First Element is 0 (Smoothing on X)
  #######################################
  if orders[0] == 0:
    sigmaRange,derx_1D = gaussianMask1D(sigma=sigma, order = orders[0]) 
    #if second element = 0, smoothing on Y
    if orders[1]==0:
      sigmaRange,dery_1D = gaussianMask1D(sigma=sigma, order = orders[1])
      outim = cv2.sepFilter2D(im,-1,derx_1D,dery_1D)
      return outim.astype("float64")
    #if second element = 1, 1st Derivative on Y
    elif orders[1]==1:
      sigmaRange,dery_1D = gaussianMask1D(sigma=sigma, order = orders[1])
      outim = cv2.sepFilter2D(im,-1,derx_1D,dery_1D)
      return outim.astype("float64")
    #if second element = 2, 2nd Derivative on Y
    elif orders[1]==2:
      sigmaRange,dery_1D = gaussianMask1D(sigma=sigma, order = orders[1])
      outim = cv2.sepFilter2D(im,-1,derx_1D,dery_1D)
      return outim.astype("float64")
  #######################################
  #If first element is 1 (1st Derivative on X)
  #######################################
  elif orders[0] == 1:
    sigmaRange,derx_1D = gaussianMask1D(sigma=sigma, order = orders[0]) 
    #if second element = 0, smoothing on Y
    if orders[1]==0:
      sigmaRange,dery_1D = gaussianMask1D(sigma=sigma, order = orders[1])
      outim = cv2.sepFilter2D(im,-1,derx_1D,dery_1D)
      return outim.astype("float64")
    #if second element = 1, 1st Derivative on Y
    elif orders[1]==1:
      sigmaRange,dery_1D = gaussianMask1D(sigma=sigma, order = orders[1])
      outim = cv2.sepFilter2D(im,-1,derx_1D,dery_1D)
      return outim.astype("float64")
    #if second element = 2, 2nd Derivative on Y
    elif orders[1]==2:
      sigmaRange,dery_1D = gaussianMask1D(sigma=sigma, order = orders[1])
      outim = cv2.sepFilter2D(im,-1,derx_1D,dery_1D)
      return outim.astype("float64")
  #######################################
  #If first element is 2 (2nd Derivative on X)
  #######################################
  elif orders[0] == 2:
    sigmaRange,derx_1D = gaussianMask1D(sigma=sigma, order = orders[0]) 
    #if second element = 0, smoothing on Y
    if orders[1]==0:
      sigmaRange,dery_1D = gaussianMask1D(sigma=sigma, order = orders[1])
      outim = cv2.sepFilter2D(im,-1,derx_1D,dery_1D)
      return outim.astype("float64")
    #if second element = 1, 1st Derivative on Y
    elif orders[1]==1:
      sigmaRange,dery_1D = gaussianMask1D(sigma=sigma, order = orders[1])
      outim = cv2.sepFilter2D(im,-1,derx_1D,dery_1D)
      return outim.astype("float64")
    #if second element = 2, 2nd Derivative on Y
    elif orders[1]==2:
      sigmaRange,dery_1D = gaussianMask1D(sigma=sigma, order = orders[1])
      outim = cv2.sepFilter2D(im,-1,derx_1D,dery_1D)
      return outim.astype("float64")

  else:
    ('error in order of derivative')

def gradientIM(im,sigma):
  # compute order-1 derivatives on boy X and Y directions
  orders=[1,0] #First Deriv on X
  dx = my2DConv(im, sigma, orders)
  orders=[0,1] #First Deriv on Y
  dy = my2DConv(im, sigma, orders)
  return dx,dy

def laplacianG(im,sigma):
  #Calculating the Laplacian
  tmpX = []
  tmpY = []
  orders=[2,0] #Second Deriv on X
  tmpX = my2DConv(im, sigma, orders)
  orders=[0,2] #Second Deriv on Y
  tmpY = my2DConv(im, sigma, orders)
  #This is already normalized because of the distributive property of
  #second derivative normalization
  imLaplac = np.add(tmpX,tmpY)
  return imLaplac

def magnitudeGrad(dx,dy):
  magIm = np.sqrt(np.add(np.power(dx,2),np.power(dy,2)))
  return magIm

def orientationGrad(dx,dy):
  a = np.arctan2(dy,dx)*(180 / np.pi) #Orientation is calculated and converted to degrees
  a[a < 0] = a[a < 0] + 360 #to compensate for the negative values to make them correct
  print("Gradient Done")
  orientIm = a
  return orientIm

def pyramidGauss(im,sizeMask=7, nlevel=4,flagInterp=cv2.INTER_LINEAR,sigma=1):
  #4 niveles adicionales a la imagen original
  #sizeMask es el tamaño del filtro que vamos a pasar

  #Asignamos a la memoria la imagen original
  pyramidMemory = [im]

  orders=[0,0] #Second Deriv on X and Y
  
  #For each of the desired levels will do smoothing, scaling down and appending
  for x in np.arange(nlevel):
    #Smoothing
    blurIm = my2DConv(pyramidMemory[x], sigma, orders)
    rows = blurIm.shape[0]
    cols = blurIm.shape[1]
    #Sampling down by half
    samplingRows = np.ceil(rows/2).astype(int)
    samplingCols =  np.ceil(cols/2).astype(int)
    redux = cv2.resize(blurIm.astype('float64'),(samplingCols, samplingRows) ,flagInterp)
    #Append the new image on the memory
    pyramidMemory.append(redux)

  return pyramidMemory

def pyramidLap(im, sizeMask=0,nlevel=4,flagInterp=cv2.INTER_LINEAR,sigma=1):
  #compute the Gaussian Pyramid
  pyramidMemory = pyramidGauss(im,sizeMask, nlevel,flagInterp,sigma) #Sizemask 7x7 and Number of levels 4

  #Rescaling blur (making image bigger again after blur from the gaussian pyramid)
  rescaleBlur = []
  cont = 0

  #We skip first image because that's the original
  for i in pyramidMemory[1:]:
    #We select the size of the image inmediatly bigger
    scaleRows = pyramidMemory[cont].shape[0]
    scaleCols = pyramidMemory[cont].shape[1]
    tmp = cv2.resize(i.astype('float64'),(scaleCols,scaleRows) ,flagInterp)
    #Append results
    rescaleBlur.append(tmp)
    cont = cont + 1
 
  # Go through the Gaussian Pyramid computing the differences
  laplacMemory = []
  blurCount = 0

  #For 
  for pyr in pyramidMemory:
    #Accedemos a las imagenes escaladas y calculamos la diferencia
    try:
      tmp = np.subtract(pyr,rescaleBlur[blurCount])
      laplacMemory.append(tmp)
      blurCount = blurCount + 1
    #Al acabar imprimimos "Done"
    except:
      print("Done")
  # Finishing the pyramid
  return laplacMemory,pyramidMemory

def reconstructIm(pyL,pyramidMemory,flagInterp):
  # pyL is a vector hosting the Laplacian Pyramid
  # Go top-bottom through pyL to reconstruct the image
  #Rescaling blur (making image again after blur)

  #Initial value is the last one of the gaussian pyramid
  imInicial = pyramidMemory[len(pyramidMemory)-1]

  #We flip the Order to start with the last of the laplacian
  laplacMemory = np.flip(pyL)

  tmp = imInicial
  restoredImg = []
  #On each iteration of the laplacian memory we add and store
  for x in np.arange(len(laplacMemory)):
    #Calculate the image
    scaleRows = laplacMemory[x].shape[0]
    scaleCols = laplacMemory[x].shape[1]
    tmp = cv2.resize(tmp.astype('float32'),(scaleCols,scaleRows) ,flagInterp)
    tmp = np.add(laplacMemory[x],tmp)
    restoredImg.append(tmp)

  #We flip to see the higher image first
  restoredImg = np.flip(restoredImg)

  return restoredImg