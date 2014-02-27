import dicom, pylab
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
    A = dicom.read_file("knee1.dcm").pixel_array
    B = dicom.read_file("knee2.dcm").pixel_array

    I_AB = MutualInformation(A,B)
    I_BA = MutualInformation(B,A)
    I_AA = MutualInformation(A,A)
    print I_AB
    print I_BA
    print I_AA
    
    JointHistogram = ShannonEntropy(A, B, GetHistogram = True)
    pylab.imshow(JointHistogram)
    showPlot(max(JointHistogram.shape))

if __name__ == "__main__":
    main()