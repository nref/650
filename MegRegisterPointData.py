# Meg Drouhard
# 2/1/14
# Basic image registration for 2D datasets using principle axes analysis

import pylab
from numpy import *
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib as mpl
import copy
import math

# Calculate angle between 2 vectors
def angle(v1, v2):
    return math.acos(dot(v1,v2)/(length(v1)*length(v2)))

# Calculates length of a vector
def length(v):
    return math.sqrt(dot(v, v))

# Performs basic image registration on 2D lists of tuples (point data)
def register(obj1, obj2):
    print "\n"
    # calculate centroids
    centroid1 = (sum(obj1[0])/len(obj1[0]), sum(obj1[1])/len(obj1[1]))
    centroid2 = (sum(obj2[0])/len(obj2[0]), sum(obj2[1])/len(obj2[1]))
    print "Centroid1: " + str(centroid1)
    print "Centroid2: " + str(centroid2)

    # Shift data to move centroids to origin
    shifted_obj1 = copy.deepcopy(obj1)# obj1 shifted so centroid is at origin
    shifted_obj2 = copy.deepcopy(obj2)# obj2 shifted so centroid is at origin

    for i in range(len(obj1[0])):
        shifted_obj1[0][i] = obj1[0][i] - centroid1[0] 
        shifted_obj1[1][i] = obj1[1][i] - centroid1[1] 
        shifted_obj2[0][i] = obj2[0][i] - centroid2[0] 
        shifted_obj2[1][i] = obj2[1][i] - centroid2[1] 

    print "\n"
    print "shifted obj1:"
    print shifted_obj1
    print "shifted obj2:"
    print shifted_obj2

    # Calculate covariance matrices
    cov1 = cov(shifted_obj1[0],shifted_obj1[1])
    cov2 = cov(shifted_obj2[0],shifted_obj2[1])

    # Calculate eigenvectors and eigenvalues and sort
    eigVals1, eigVecs1 = linalg.eig(cov1)
    eigVals2, eigVecs2 = linalg.eig(cov2)

    #s_eigVals1 = np.sort(eigVals1)
    #s_eigVecs1 = eigVecs1[eigVals1.argsort()] 

    # Sort eigenvalues and eigenvectors in place from least-to-greatest
    sort_perm1 = eigVals1.argsort()
    eigVals1.sort()
    eigVecs1 = eigVecs1[sort_perm1]

    sort_perm2 = eigVals2.argsort()
    eigVals2.sort()
    eigVecs2 = eigVecs2[sort_perm2]

    # Calculate angle between eigenvectors 
    # http://stackoverflow.com/questions/2827393/
    theta = angle(eigVecs1[0], eigVecs2[0])    

    # Rotate data to align eigenvectors with axes
    # Rotation matrix
    R = array([[-cos(theta), sin(theta)],
               [-sin(theta), -cos(theta)]])

    
    # Register data (rotate 2nd object to align with 1st)
    imageSoln = R.dot(shifted_obj2)

    print "\nRegistered Data:"
    print "shifted obj1:"
    print shifted_obj1
    print "solution (registered obj2):"
    print imageSoln
    


    # Draw original data
    pylab.plot(obj1[0], obj1[1], 'bo')
    pylab.plot(obj2[0], obj2[1], 'ro')

    # Draw shifted data
    #pylab.plot(shifted_obj1[0], shifted_obj1[1], 'bo')
    #pylab.plot(shifted_obj2[0], shifted_obj2[1], 'ro')

    # Draw registered data
    pylab.plot(imageSoln[0], imageSoln[1], 'go')
    



def main():
    #Same point data as Doug's example
    cluster1 = array([[0.5,0.5],[0.5,1.0],[1.0, 0.5],[1.0,1.0],[1.0,1.5],
                 [1.5,1.0]]).T 
    cluster2 = array([[-0.5,0.5],[-1.0,0.5],[-0.5,1.0],[-1.0,1.0],[-1.5,1.0],
                  [-1.0,1.5]]).T

    print "cluster1:"
    print cluster1
    print "cluster2:"
    print cluster2

    # Register 2 sets of data
    register(cluster1, cluster2)

    # Draw origin
    pylab.axvline(0,-100,100, color='k')
    pylab.axhline(0,-100,100, color='k')

    # Axes' bounds
    s = 2
    pylab.axis([-s,s,-s,s])

    pylab.xlabel("X")
    pylab.ylabel("Y")
    pylab.show()

if __name__ == "__main__":
    main()

