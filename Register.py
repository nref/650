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
from ImgLib import *

def showPlot(plotBoundary):
    # Draw origin
    pylab.axvline(0,-100,100, color='k')
    pylab.axhline(0,-100,100, color='k')
    
    # Axes' bounds
    s = plotBoundary
    pylab.axis([0,s,0,s])
    
    pylab.xlabel("X")
    pylab.ylabel("Y")
    pylab.show()

def main():
    # Get images
    knee1 = dicom.read_file("knee1.dcm").pixel_array
    knee2 = dicom.read_file("knee2.dcm").pixel_array
    knee1T, knee2TR = imgRegister(knee1, knee2, silent = False)
    
#     knee1_cpy = imgUnpad(imgTranslate(imgRotate(imgPad(knee1), 5.0), 4, 4))
#     A, B = imgRegister(knee1, knee1_cpy, silent = False)
#     print I(A, B)

    showPlot(max(multiply(1,knee1.shape)))

if __name__ == "__main__":
    main()