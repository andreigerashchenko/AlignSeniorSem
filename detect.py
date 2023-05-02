import cv2
import numpy as np
from scipy.interpolate import splprep, splev
from horizonfinder import score_candidate, find_horizon_point

# Define minimum and maximum average height of the horizon spline (in percent of the image height, 0.0-1.0 = 0%-100%)
MIN_HEIGHT = 0.3
MAX_HEIGHT = 0.7

# Define weights for scoring
LENGTH_WEIGHT = 1.0
SMOOTHNESS_WEIGHT = 1.0
LINEARITY_WEIGHT = 1.0

# Load the image
img = cv2.imread(r'examples\R0011200.JPG')

point_x, point_y, point_type = find_horizon_point(img, LENGTH_WEIGHT, SMOOTHNESS_WEIGHT, LINEARITY_WEIGHT, MIN_HEIGHT, MAX_HEIGHT)

print(point_x, point_y, point_type)

# Draw the horizon point in blue
cv2.circle(img, (point_x, point_y), 25, (255, 0, 0), -1)

# Resize the image if it's larger than 1280x720
if img.shape[0] > 720 or img.shape[1] > 1280:
    img = img.copy() # Make a copy of the image so we don't modify the original
    scale_factor = min(1280 / img.shape[1], 720 / img.shape[0])
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

# Show the image
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
