import dicom, pylab
from ImgLib import *

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
#    A = dicom.read_file("knee1_cpy.dcm").pixel_array
#    B = dicom.read_file("knee1_rot_5_deg_shift_10_x_10_y.dcm").pixel_array

    A = dicom.read_file("knee1.dcm").pixel_array
#    B = dicom.read_file("knee2.dcm").pixel_array
    B = imgUnpad(imgTranslate(imgRotate(imgPad(A), 5.0), 4, 4))

    I_AA = I(A,A)
    I_BB = I(B,B)
    I_AB = I(A,B)
    I_BA = I(B,A)

    print "I(A,A): %f" % I_AA
    print "I(B,B): %f" % I_BB
    print "I(A,B): %f" % I_AB
    print "I(B,A): %f" % I_BA

#    print "Registering image A to B with Principal Axes"
#    A_registered, B_registered = imgRegister(A, B, silent = False, showPlots = True)
#    I_AB_reg = I(A_registered, B_registered)
#    showPlot(max(multiply(1,A.shape)))
#    print "I(A_registered, B): %f" % I_AB_reg
#    
#    print "Showing histogram"
#    JointHistogram = imgDistributions(A_registered, B_registered, GetHistogram = True)
#    pylab.imshow(JointHistogram)
#    showPlot(max(JointHistogram.shape))

    print "Registering image A to B with MutualInformation"
    A_registered, B_registered = MIRegister(A, B)
    I_AB_reg = I(A_registered, B_registered)
    print "I(A_registered, B): %f" % I_AB_reg

    print "Showing histogram"
    JointHistogram = imgDistributions(A_registered, B_registered, GetHistogram = True)
    pylab.imshow(JointHistogram)
    showPlot(max(JointHistogram.shape))

    print "Showing registered images"
    pylab.imshow(A_registered, cmap=pylab.cm.bone, alpha=0.5)
    pylab.imshow(B_registered, cmap=pylab.cm.bone, alpha=0.5)
#    pylab.imshow(abs(A_registered-B_registered), cmap=pylab.cm.bone, alpha=0.5)
    showPlot(max(multiply(1,A.shape)))

if __name__ == "__main__":
    main()