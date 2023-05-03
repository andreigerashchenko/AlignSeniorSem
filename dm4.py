'''
Purpose: Straightens the horizon of DM's 360 KAPs, which
         are taken with a Ricoh Theta S hanging upside down
         from the kite line, a little below the kite.

Usage: python3 dm4.py <path/to/input/360image>

Requires: opencv, numpy, matplotlib, and pyEquirectRotate from
          https://github.com/BlueHorn07/pyEquirectRotate

2022-Jun-04 (DM): First usable version.

'''
import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from equirectRotate import EquirectRotate, pointRotate

if __name__ == '__main__':

  ###################################################################
  # open source image and display it to get lowest earth-sky horizon

  src_path = input('Enter path to input 360 image: ')
  
#   src_path = sys.argv[1]
  opfile = os.path.splitext(src_path)[0]+'_f.jpg'

  print('\n *****************************************************')
  print(' EITHER find the lowest point on the Earth-sky horizon and')
  print(' press 1 (one) to select the coordinates of that point,')
  print(' OR find the highest point on the Earth-sky horizon and')
  print(' press 9 (nine) to select the coordinates of that point.')
  print(' Then press q to quit and return to main program.\n')

  def press(event):
    global ix, iy
    ix = 0
    iy = 0
    if event.key == '1':
        ix, iy = event.xdata, event.ydata
        print(' Lowest horizon cursor x, y: ',ix, iy)
    if event.key == '9':
        ix, iy = event.xdata, -event.ydata
        print(' Highest horizon cursor x, y: ',ix, -iy)
    if event.key == 'q':
        fig.canvas.mpl_disconnect(pid)

    return ix, iy

  fig = plt.figure()
  img = mpimg.imread(src_path)
  imgplot = plt.imshow(img)
  plt.grid()
  pid = fig.canvas.mpl_connect('key_press_event', press)
  plt.show()
  ###################################################################

  ###################################################################
  print('\n Now rotating the image to straighten the horizon.')
  src_image = cv2.imread(src_path)
  h, w, c = src_image.shape
  print("\n Input file's height, width, colors =", h,w,c)

  # Do a 'yaw' rotation such that ix position earth-sky horizon is 
  # at the middle column of the image. Fortunately for an equirectangular
  # image, a yaw is simply sliding the image horizontally, and is done very
  # fast by np.roll.
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


  print('\n Doing the final rotation (pitch =',str(f'{myP:.2f}'),
          'deg). This can take a while ...')
  # rotate (yaw, pitch, roll)
  equirectRot = EquirectRotate(h, w, (myY, myP, myR))
  rotated_image = equirectRot.rotate(src_image)
  ###################################################################

  final_image = cv2.rotate(rotated_image, cv2.ROTATE_180)

  cv2.imwrite(opfile, final_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
  print('\nWrote output file: ', opfile)
  print('Done.')

