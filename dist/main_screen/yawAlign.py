'''
Purpose: The user provide the path to the directory where a bunch of 360-deg
         equirectangular .jpg images are located. The code opens each .jpg
         image in a window and waits for the user to click somewhere on that
         image. Once the user clicks on any location on as input image, that 
         column (longitude) is 'yaw'-ed to become the central meridian of the
         output image.

Usage: python3 yawAlign.py <path/to/input/360images-folder>

Output: for each .jpg image in path/to/input/360images-folder/*.jpg the code
        produces path/to/input/360images-folder/*_aligned.jpg

Requires: numpy, matplotlib, opencv

2022-Jun-21: First usable version (DM).
2022-Sep-04: Cleaning up the code a bit (DM).
'''


import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Get cursor coordinates from key press event
def press(event):
    global ix, iy
    if event.key == '1':
        ix, iy = event.xdata, event.ydata
        print(' cursor x, y: ',ix, iy)
    if event.key == 'q':
        fig.canvas.mpl_disconnect(pid)
    return ix, iy


if __name__ == '__main__':

  src_path = sys.argv[1]

  for x in os.listdir(src_path):
      if x.endswith(".jpg"):
          ipfile = os.path.join(src_path, x)
          opnam = os.path.splitext(x)[0] + '_aligned.jpg'
          opfile = os.path.join(src_path, opnam)

          # open source image and display it to choose the longitude 
          # to turn into the central meridian
          fig = plt.figure()
          img = mpimg.imread(ipfile)
          imgplot = plt.imshow(img)
          plTitle = "File: " + ipfile + "\n"
          plTitle+= "Find the point of interest and "
          plTitle+= "press 1 (one) to select the point's coordinates.\n"
          plTitle+= "Then press q to quit and return to main program."
          plt.title(plTitle)
          plt.grid()
          pid = fig.canvas.mpl_connect('key_press_event', press)
          plt.show()

          # Do a 'yaw' rotation such that ix position earth-sky horizon is 
          # at the middle column of the image. Fortunately for an 
          # equirectangular image, a yaw is simply sliding the image 
          # horizontally, and is done very fast by np.roll.
          print('\n Now applying the required yaw.')
          src_image = cv2.imread(ipfile)
          h, w, c = src_image.shape

          shiftx=int(w/2 - ix)
          src_image = np.roll(src_image, shiftx, axis=1) 

          # Write the output image
          cv2.imwrite(opfile, src_image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
          print('\nWrote output file: ', opfile)


print('All done.')

