# ImgLib
# Image processing library
# CS650 Image Processing Dr. Jens Gregor
# Spring 2014
# Doug Slater
# mailto:cds@utk.edu

import dicom, pylab
from numpy import *
from scipy import ndimage, optimize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import copy
import sys, os

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
    return degrees(math.acos(dot(v1,v2)/(length(v1)*length(v2))))

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

def MITryTransformations( (x, y, theta), A, B):

    # Restrict the possible transformations
#    if abs(x) > 10 or abs(y) > 10 or abs(theta) > 15:
#        return float("inf")

    sys.stdout.write("Trying %f %f %f: " % (x, y, theta))

    A_Translated_Rotated = imgTranslate(imgRotate(A, theta), x, y)
    MI = -I(A_Translated_Rotated, B)

    sys.stdout.write("%f\n" % MI)
    return MI

def MIRegister(A, B, silent = True):
    
    # Initial guess: 1x1 pixel x,y translation and 1 deg rotation
    guess = [10, 10, 10]
    
    # Minimize with Powell's method, which doesn't require derivative,
    # but only the value of the objective function.
    x, y, theta = optimize.fmin_powell(MITryTransformations,
                                       guess,
                                       args=(A, B),
                                       xtol = 0.001,
                                       ftol = 0.001,
                                       disp = silent)
    
    # Return A with optimal transformation
    #A_optimal = imgRotate(imgTranslate(A, x, y), theta)
    A_optimal = imgTranslate(imgRotate(A, theta), x, y)
    return A_optimal, B

def I(A, B, UseKullback = True):
    """
    Compute the mutual information I(A,B) of two 2D images A and B
    using Kullback Leibler or Pluim et al. 2003 eq 6
    """
    if UseKullback is True:  I_AB = KullbackLeibler(A, B)
    else:                    I_AB = PluimEq6(A, B)
    
    return I_AB # I(A,B)

def JointHistogram(A, B):
    """
    Return the 2D histogram and edges for two images A and B
    """
    
    # 2D histogram gives the counts in each cell
    hist, A_edges, B_edges = \
        histogram2d(A.ravel(), B.ravel(), bins=min(shape(A))/20)

    return hist, A_edges, B_edges

def PluimEq2(P):
    """
    Pluim 2003 p.2 equation (2)
    http://www.cs.jhu.edu/~cis/cista/746/papers/mutual_info_survey.pdf
    """
    H = -sum([p * math.log(p,2) for p in P.flatten() if p != 0])
    
    return H

def PluimEq6(A, B):
    """
    Pluim 2003 p.3 equation (6)
    http://www.cs.jhu.edu/~cis/cista/746/papers/mutual_info_survey.pdf
    """

    I_AB = H(A) + H(B) - H(A,B)
    return I_AB

def H(A, B = None):
    return Shannon(A, B)

def Shannon(A, B = None):
    """
    Compute the Shannon entropy H(A) of a single image A
    Or compute the joint Shannon entropy H(A,B) of two images A and B
    Credit http://brainacle.com/calculating-image-entropy-with-python-how-and-why.html
    """
    
    hist, bins = histogram(A)          # Probability distributions of grey values
    P = hist / hist.sum()              # P(A) or P(B|A) if B provided
    
    if B is not None:
        histAB, A_edges, B_edges = JointHistogram(A, B)
        P = histAB / histAB.sum()    # Joint Probability Distribution P(A, B)
    
    H = PluimEq2(P)
    
    return H

def imgDistributions(A, B, GetHistogram = False):
    """
    Get the joint probability distribution and individual
    probability distributions of two images A and B
    Credit http://stackoverflow.com/questions/9002715
    """

    hist, A_edges, B_edges = JointHistogram(A, B)

    # Get the joint probability distribution the images' grey values
    # Divide by the total to get the probability of each cell
    P_AB = hist / hist.sum()      # P(A,B)

    # Sum over the axes to get their marginal entropy
    P_A = hist.sum(1)[:,newaxis]  # P(A) (nx1)
    P_B = hist.sum(0)[newaxis,:]  # P(B) (1xn)

    if GetHistogram: return hist
    return P_AB, P_A, P_B

def KullbackLeibler(A, B):
    """
    Compute the Kullback-Leibler distance for two 2D images A and B
    Credit http://stackoverflow.com/questions/9002715
    """
    seterr(all = 'ignore')    # Suppress log div by zero warning
    
    P_AB, P_A, P_B = imgDistributions(A, B) # P(A,B), P(A), P(B)
    
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
    """ Rotate a 2D image by theta degrees """
    return ndimage.interpolation.rotate(image, theta, reshape=False)

def imgTranslate(image, x, y):
    """ Shift a 2D image by x, y pixels """
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

def imgRegister(img1, img2, silent = True, showPlots = True, drawDifference = True):
    """
    Register img2 to img1 using principal axes
    1st return value is img1 translated to center
    2nd return value is img2 translated to center and rotated to img1
    """
    
    # Redirect stdout to /dev/null
    stdout = sys.stdout;
    if silent:
        f = open(os.devnull, 'w')
        sys.stdout = f
    
    # Pad image to enable rotation without clipping
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
    # TODO: fix this
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
    
    # Hack alert: I don't know how to order the eigenvalues.
    # Currently they are coming out orthonormal.
    # I 'fix' this by normalizing rotation to under 90 degrees.
    # a.k.a This algorithm may rotate the wrong direction for
    # rotations >= 45 degrees
    theta = normalizeDegrees(theta)
    
    sys.stdout.write("\nRotating image %.4f degrees\n" % (theta))
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
 
    if showPlots:
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
    
        if drawDifference:
            pylab.imshow(abs(img1T_unpadded-img2TR_unpadded), cmap=pylab.cm.bone, alpha=0.5)
        else:
            # Draw the unregistered 1st and registered 2nd image
            pylab.imshow(img1T_unpadded, cmap=pylab.cm.bone, alpha=0.5)
            #pylab.imshow(img2T_unpadded, cmap=pylab.cm.bone, alpha=0.5)
            pylab.imshow(img2TR_unpadded, cmap=pylab.cm.bone, alpha=0.5)
        
        

    # Restore stdout
    if silent: sys.stdout = stdout
    
    return img1T, img2TR