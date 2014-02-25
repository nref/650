# slaterImgRegister.py
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
import copy
import sys
from SlaterImgLib import *

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
    # Get images
    knee1 = dicom.read_file("knee1.dcm").pixel_array
    knee2 = dicom.read_file("knee2.dcm").pixel_array
    
    knee1T, knee2TR = imgRegister(knee1, knee2)
    
    showPlot(max(multiply(1,knee1.shape)))

if __name__ == "__main__":
    main()