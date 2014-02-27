# SlaterImgLib
# Image processing library
# CS650 Image Processing Dr. Jens Gregor
# Spring 2014
# Doug Slater
# mailto:cds@utk.edu

import dicom, pylab
from numpy import *
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import copy
import sys

def normalizeRadians(radians):
    """ Normalize an angle to pi/2 radians """
    return radians % (pi/2)
	
def normalizeDegrees(degrees):
    """ Normalize an angle to 90 degrees """
    return degrees % 90
	 
def radians(degrees):
    """ Convert an angle in degrees to radians """
    return degrees*(pi/180)

def degrees(radians):
    """ Convert an angle in radians to degrees """
    return radians*(180/pi)

def angle(v1, v2):
    """
    Calculate angle between 2 vectors
    http://stackoverflow.com/questions/2827393/
    """
    return math.acos(dot(v1,v2)/(length(v1)*length(v2)))

def length(v):
    """ 
    Calculate length of a vector
    Credit Meg Drouhard
    """
    return math.sqrt(dot(v, v))

def drawScaledEigenvectors(X,Y, eigVectors, eigVals, theColor='k'):
    """ Draw scaled eigenvectors starting at (X,Y)"""
    
    # For each eigenvector
    for col in xrange(eigVectors.shape[1]):
        
        # Draw it from (X,Y) to the eigenvector length
        # scaled by the respective eigenvalue
        pylab.arrow(X,Y,eigVectors[0,col]*eigVals[col],
                    eigVectors[1,col]*eigVals[col],
                    width=0.01, color=theColor)

def ShannonEntropy(A, B = None, GetHistogram = False):
    """ 
    Compute the conditional Shannon entropy H(B|A) of two images A and B
    Credit http://stackoverflow.com/questions/9002715
    """
    
    if B is None:
        
        """
        Compute the Shannon entropy H(B) of a single image B
        Credit http://brainacle.com/calculating-image-entropy-with-python-how-and-why.html
        """
        # Get probability distribution of grey values
        histogram, bins = histogram(B)
        P_B = histogram / histogram.sum()

        # Pluim 2003 p.2 equation (2)
        # http://www.cs.jhu.edu/~cis/cista/746/papers/mutual_info_survey.pdf
        H = -sum([p * math.log(p,2) for p in P_B if p != 0])
        return H
    
    # 2D histogram gives the counts in each cell
    H, A_edges, B_edges = \
        histogram2d(A.ravel(), B.ravel(), bins=min(shape(A))/20)

    # Get the joint probability distribution the images' grey values
    # Divide by the total to get the probability of each cell
    P_AB = H/H.sum()    # P(A,B)

    # Sum over the axes to get their marginal entropy
    P_A = H.sum(1)[:,newaxis]  # P(A) (nx1)
    P_B = H.sum(0)[newaxis,:]  # P(B) (1xn)

    if GetHistogram: return H
    return P_AB, P_A, P_B

def MutualInformation(A, B):
    """
    Compute the mutual information I(A,B) of two 2D images A and B
    """
    I_AB = KullbackLeibler(A, B)  # I(A,B)
    return I_AB
def KullbackLeibler(A, B):
    """
    Compute the Kullback-Leibler distance for two 2D images A and B
    Credit http://stackoverflow.com/questions/9002715
    """
    seterr(all = 'ignore')    # Suppress log div by zero warning
    
    P_AB, P_A, P_B = ShannonEntropy(A,B) # P(B,A), P(A), P(B)
    
    # Pluim 2003 p.4 eq. (7)
    # http://www.cs.jhu.edu/~cis/cista/746/papers/mutual_info_survey.pdf
    KL = P_AB * log(P_AB/(P_A*P_B)) # Get information contribution
    I_AB = KL[~isnan(KL)].sum()     # Filter nans and sum
    
    seterr(all = 'warn')
    return I_AB                     # I(A,B)

def imgCov(data):
    """ 
    Compute the x-mean, y-mean, and
    return the cov matrix of a 2D image.
    Credit http://stackoverflow.com/questions/9005659/
    """
    
    def raw_moment(data, iord, jord):
        nrows, ncols = data.shape
        y, x = mgrid[:nrows, :ncols]
        data = data * x**iord * y**jord
        return data.sum()
    
    data_sum = data.sum()
    m10 = raw_moment(data, 1, 0)
    m01 = raw_moment(data, 0, 1)
    x_bar = m10 / data_sum
    y_bar = m01 / data_sum
    u11 = (raw_moment(data, 1, 1) - x_bar * m01) / data_sum
    u20 = (raw_moment(data, 2, 0) - x_bar * m10) / data_sum
    u02 = (raw_moment(data, 0, 2) - y_bar * m01) / data_sum
    cov = array([[u20, u11], [u11, u02]])
    return cov

def imgCentroid(im):
    """ 
    Get the centroid of a 2D image
    http://code.activestate.com/lists/python-image-sig/5121/
    """
    sx = sy = n = 0
    x0, y0 = 0, 0
    x1, y1 = im.shape
    for y in range(y0, y1):
        for x in range(x0, x1):
            if im[x, y]:
                sx += x
                sy += y
                n += 1
    return float(sx) / n + 0.5, float(sy) / n + 0.5

def imageApplyHarshWeightingScheme(image, threshold=200):
    """ 
    Apply weighting to a 2D image, i.e. scale color values
    Gate all values above the threshold to the maximum value
    """
    
    imageW  = copy.deepcopy(image)       # weighted image
    max     = imgGetMaxPixelValue(image)
    
    for row in xrange(image.shape[0]):
        for col in xrange(image.shape[1]):
            
            if image[row][col] > threshold:
                imageW[row][col] = max
    return imageW

def imgGetMaxPixelValue(image):
    """ Return the maximum pixel value in an image """
    (x,y) = unravel_index(image.argmax(), image.shape)
    return image[x][y]

def imgMadness(image):
    """ 
    Get the X-pixels, Y-pixels, and scaled RGB values from a 2D image
    """
    
    # Get maximum pixel value for scaling
    max = float(imgGetMaxPixelValue(image))
    
    dims        = image.shape
    pixelsX     = zeros(dims, dtype=int)
    pixelsY     = zeros(dims, dtype=int)
    pixelRGB    = zeros(dims, dtype=float)
    
    for x in xrange(dims[0]):
        for y in xrange(dims[1]):
            pixelsX[x][y]   = x
            pixelsY[x][y]   = y
            pixelRGB[x][y]  = image[x][y]/(max/255)  # Scale 0 to 255

    return (pixelsX, pixelsY, pixelRGB)

def imgPad(image):
    """
    Apply black padding to a 2D image. Doubles both dimensions and
    places original image at center
    """
    
    newImg = zeros(multiply(image.shape,2), dtype=int)
    x,y = newImg.shape
	
    x1, x2 = x/4, 3*x/4
    y1, y2 = y/4, 3*y/4
	
    for i in xrange(x):
        for j in xrange(y):
            
            if (i >= x1 and i < x2 and j >= y1 and j < y2):
                newImg[i][j] = image[i-x1][j-y1]
    return newImg

def imgUnpad(image):
    """
    Inverse operation of imgPad(). Crop first and last quarter
    of a 2D image on both axes.
    """
    
    x,y = image.shape
    x1, x2 = x/4, 3*x/4
    y1, y2 = y/4, 3*y/4

    return image[x1:x2, y1:y2]

def imgRotate(image, theta):
    """ Rotate a 2D image """
        
    # Convert image data to point data which indexes color values
    pixelsX, pixelsY, pixelRGB = imgMadness(image)
    xOrds = pixelsX.flatten()
    yOrds = pixelsY.flatten()
    
    imageCoords = []
    for i in xrange(len(xOrds)):
        imageCoords.append([xOrds[i],yOrds[i]])
    imageCoords = array(imageCoords).T

    # Rotate the point data
    R = array([[cos(theta), -sin(theta)],
               [sin(theta),  cos(theta)]])
    imageR = R.dot(imageCoords)

    # Convert point data back to displayable image
    displayableImageR = zeros(image.shape, dtype=int)

    i=0
    for x in xrange(image.shape[0]):
        for y in xrange(image.shape[1]):
            RGB_x = int(imageR[0][i])
            RGB_y = int(imageR[1][i])
            try:
                displayableImageR[x][y] = pixelRGB[RGB_x][RGB_y]
            except IndexError:  # These pixels will be clipped
                pass
            i+=1
                                                                  
    return displayableImageR
    #return ndimage.interpolation.rotate(image, theta*180/pi)

def imgTranslate(image, x, y):
    """ Shift a 2D image by x,y """
    return ndimage.interpolation.shift(image, (x,y))

def imgTranslateCentroidToCenter(image):
    """ Translate a centroid of a 2D image to its geometric center """
    return imgTranslateToOrigin(image, toCenter=True)

def imgTranslateToOrigin(image, xOrig=0, yOrig=0, toCenter=False):
    ''' Shift the centroid of an image to the origin 
    	or to the specified coordinate
    	or to the center of the image.
    '''
    centroid = imgCentroid(image)
        
    center = ()
    if toCenter is True:
    	center = (image.shape[0]/2, image.shape[1]/2)
    	xOrig,yOrig = center
    	
    return imgTranslate(image, xOrig-centroid[0], yOrig-centroid[1])

def imgRegister(img1, img2):
    """
    Register img2 to img1
    1st return value is img1 translated to center
    2nd return value is img2 translated to center and rotated to img1
    """
    
    # Pad image so that we can do imaging without clipping
    sys.stdout.write("Padding image...\n")
    sys.stdout.flush()
    
    img1p = imgPad(img1)
    img2p = imgPad(img2)
    
    sys.stdout.write("Translating centroids...\n")
    sys.stdout.flush()
    
    img1T = imgTranslateCentroidToCenter(img1p)
    img2T = imgTranslateCentroidToCenter(img2p)
    
    sys.stdout.write("Applying weighting...\n")
    sys.stdout.flush()
    img2T_w = imageApplyHarshWeightingScheme(img1T) # Apply weighting scheme

    sys.stdout.write("Computing covariance matrices...\n")
    sys.stdout.flush()
    
    # Covariance matrices
    cov1 = imgCov(imgTranslateToOrigin(img1))
    cov2 = imgCov(imgTranslateToOrigin(img2))
    covW = imgCov(img2T_w)
    
    sys.stdout.write("Computing eigenvectors, eigenvalues...\n")
    sys.stdout.flush()
    
    # Eigenvalues, eigenvectors
    eigVals1, eigVecs1		= linalg.eig(cov1)
    eigVals2, eigVecs2		= linalg.eig(cov2)
    eigValsW, eigVecsW		= linalg.eig(covW) # Eigenvectors of weighted image

    # Normalize eigenvalues to unit length
    eigVals1 = divide(eigVals1,max(eigVals1))
    eigVals2 = divide(eigVals2,max(eigVals2))
    eigValsW = divide(eigValsW,max(eigValsW))
    
    sys.stdout.write("Ordering eigenvector, eigenvalues...\n")
    sys.stdout.flush()
        
    # Vector ordering magic
    idx1		= eigVals1.argsort()[::-1]
    eigVals1	= eigVals1[idx1]
    eigVecs1	= eigVecs1[:,idx1]
    
    idx2		= eigVals2.argsort()[::-1]
    eigVals2	= eigVals2[idx2]
    eigVecs2	= eigVecs2[:,idx2]
    
    idxW		= eigValsW.argsort()[::-1]
    eigValsW	= eigValsW[idxW]
    eigVecsW	= eigVecsW[:,idxW]
    
    # Difference between weighted and nonweighted eigenvectors
    eigVecsDiff	= abs(eigVecs2 - eigVecsW)
    eigValsDiff	= abs(eigVals2 - eigValsW)
    
    set_printoptions(precision=4)
    print "\nReference Image Eigenvalues: \n%s" % eigVals1
    print "\nRotation Image Eigenvalues: \n%s" % eigVals2
    print "\nWeighted Eigenvalues: \n%s" % eigValsW
    print "\nEigenvalues - Weighted Eigenvalues: \n%s" % eigValsDiff
    print "\nReference Image Eigenvectors: \n%s" % eigVecs1
    print "\nRotation Image Eigenvectors: \n%s" % eigVecs2
    print "\nWeighted Eigenvectors: \n%s" % eigVecsW
    print "\nEigenvectors - Weighted Eigenvectors: \n%s" % eigVecsDiff

    # How much rotation will be applied
    theta = angle(eigVecs1[0], eigVecs2[0])
    
    # Hack alert: I don't know how to order the eigenvalues
    # currently they are coming out orthonormal
    # Fix by normalizing rotation to under 90 degrees
    # a.k.a This program may rotate the wrong way for
    # rotations >= 45 degrees
    theta = normalizeRadians(theta)
    
    sys.stdout.write("\nRotating image %.4f degrees\n" % (degrees(theta)))
    sys.stdout.flush()
    
    img2TR = imgRotate(img2T, theta)

    sys.stdout.write("Unpadding image...\n")
    sys.stdout.flush()
        
    img1T_unpadded = imgUnpad(img1T)
    img2T_unpadded = imgUnpad(img2T)
    img2TR_unpadded = imgUnpad(img2TR)
        
    sys.stdout.write("Plotting eigenvectors...\n")
    sys.stdout.flush()
    
    # Need to draw eigenvectors at centroids
    img1T_unpadded_centroid = imgCentroid(img1T_unpadded)
    img2TR_unpadded_centroid = imgCentroid(img2TR_unpadded)
    
    scaling = img1.shape[0]/2
    # Draw eigenvectors at centroids scaled by their eigenvalues
    drawScaledEigenvectors(	img1T_unpadded_centroid[0],
    						img1T_unpadded_centroid[1],
    						eigVecs1, eigVals1*scaling, 'b')
    drawScaledEigenvectors(	img2TR_unpadded_centroid[0],
    						img2TR_unpadded_centroid[1],
    						eigVecs2, eigVals2*scaling, 'r')
                           
    sys.stdout.write("Plotting image...\n")
    sys.stdout.flush() 
    
    # Draw the unregistered 1st and registered 2nd image
    pylab.imshow(img1T_unpadded, cmap=pylab.cm.bone, alpha=0.5)
    #pylab.imshow(img2T_unpadded, cmap=pylab.cm.bone, alpha=0.5)
    pylab.imshow(img2TR_unpadded, cmap=pylab.cm.bone, alpha=0.5)

    return img1T, img2TR