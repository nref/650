# imgRegister.py
# Register two images
# CS650 Image Processing Dr. Jens Gregor
# January 2014
# Doug Slater
# mailto:cds@utk.edu

import dicom, pylab
from numpy import *
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
from SlaterImgLib import *

def rotateData(xOrds, yOrds, center, theta, image=None):
    print "Rotating point data"
    
    coOrds = []                         # original coordinates
    coOrdsR = []                        # rotated coordinates
    
    centerR         = None              # rotated center
    imageR          = None              # rotated image
    covariance      = None              # image or data covatiance matrix
    
    # Rotation matrix
    R = array([[cos(theta), -sin(theta)],
               [sin(theta), cos(theta)]])

    # Convert x,y data to coordinates
    for i in xrange(len(xOrds)):
        coOrds.append((xOrds[i],yOrds[i]))

    # Rotate x,y data
    for (x, y) in coOrds:
        coOrdsR.append(R.dot([x,y]))
        
    covariance = cov(xOrds, yOrds)

    eigVals, eigVectors     = linalg.eig(covariance)    # Eigenvectors, eigenvalues
    eigVectorsR             = R.dot(eigVectors)         # Rotated eigenvectors
    centerR                 = R.dot(center)             # Rotated center

    set_printoptions(precision=4)   # Float precision
    print "\nUnrotated Coordinates: \n%s" % coOrds
    print "\nRotation matrix for angle " \
            + str(theta) + ": \n%s" % matrix.round(R, 4)
    print "\nRotated Coordinates: \n%s" % coOrdsR
    print "\nCovariance Matrix: \n%s" % covariance
    print "\nEigenvalues: \n%s" % eigVals
    print "\nEigenvectors: \n%s" % eigVectors
    print "\nRotated Eigenvectors: \n%s" % eigVectorsR

    # Plot original data
    pylab.plot(xOrds, yOrds, 'bo')

    # Plot rotated data
    for (xR,yR) in coOrdsR:
        pylab.plot(xR, yR, 'ro')

    # Plot eigenvectors scaled by their eigenvalues
    drawScaledEigenvectors(center[0],center[1],
                           eigVectors, eigVals, 'b')
    drawScaledEigenvectors(centerR[0],centerR[1],
                           eigVectorsR, eigVals, 'r')

def showPlot(plotBoundary):
    # Draw origin
    pylab.axvline(0,-100,100, color='k')
    pylab.axhline(0,-100,100, color='k')
    
    # Axes' bounds
    s = plotBoundary
    pylab.axis([-s,s,-s,s])
    
    pylab.xlabel("X")
    pylab.ylabel("Y")
    pylab.show()

def main():
    xOrds = array([0.5, 0.5, 1, 1, 1, 1.5])
    yOrds = array([0.5, 1, 0.5, 1, 1.5, 1])
    center = array([1, 1])
    rotateData(xOrds, yOrds, center, pi/2)

    showPlot(2)

if __name__ == "__main__":
    main()