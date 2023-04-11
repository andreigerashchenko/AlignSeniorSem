import os
import cv2
import numpy as np
from equirectRotate import EquirectRotate

def getimage():
    src_path = input('Enter path to input 360 image: ')
    opfile = os.path.splitext(src_path)[0]+'_f.jpg'

    src_image = cv2.imread(src_path)

    print("\n Input file's height, width, colors =", h,w,c)
    
    return src_image


def rotatedImage(src_image, ix,iy):

    # Do a 'yaw' rotation such that ix position earth-sky horizon is 
    # at the middle column of the image. Fortunately for an equirectangular
    # image, a yaw is simply sliding the image horizontally, and is done very
    # fast by np.roll.
    h, w, c = src_image.shape
    shiftx=int(w/2 - ix)
    src_image = np.roll(src_image, shiftx, axis=1) 

    # If iy>0 then the user selected the lowest point of the horizon.
    # After the above 'yaw', the true horizon at the middle of the image
    # is still (iy - h/2) pixels below the camera's equator. This is
    # (iy - h/2)*(180)/h degrees below the camera's equator. So rotate the
    # pitch of the yaw-ed rectilinear image by this amount to get a nearly
    # straight horizon.
    myY, myP, myR = 0, (iy - h/2)*180/h , 0

    # If iy<0 then the user actually recorded the highest point. That
    # is, the true horizon is (h/2 - |iy|) pixels above the camera's
    # equator. So rotate the pitch of the yaw-ed rectilinear image by the
    # amount -(h/2 - |iy|)*180/h to get a nearly straight horizon.
    if iy < 0 :
        myP = -(h/2 - np.abs(iy))*180/h
        
        
    # print('\n Doing the final rotation (pitch =',str(f'{myP:.2f}'),
            # 'deg). This can take a while ...')
    
    # rotate (yaw, pitch, roll)
    equirectRot = EquirectRotate(h, w, (myY, myP, myR))
    rotated_image = equirectRot.rotate(src_image)
    return rotated_image

###################################################################

def writeImage(rotated_image, opfile):
    final_image = cv2.rotate(rotated_image, cv2.ROTATE_180)
    cv2.imwrite(opfile, final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
# print('\nWrote output file: ', opfile)
# print('Done.')
    
def getCoords(highLow):
    ix = 0
    iy = 0
    if highLow:
        # ix, iy = event.xdata, event.ydata
        print(' Lowest horizon cursor x, y: ',ix, iy)
    else:
        # ix, iy = 
        print(' Highest horizon cursor x, y: ',ix, -iy)

    return ix, iy