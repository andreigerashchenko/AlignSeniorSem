import cv2
import numpy as np
from scipy.interpolate import splprep, splev

# Load the image
img = cv2.imread('examples\R0011132.JPG')

# Resize the image if it's larger than 1280x720
if img.shape[0] > 720 or img.shape[1] > 1280:
    scale_factor = min(1280 / img.shape[1], 720 / img.shape[0])
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Apply thresholding
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# Apply canny edge detection
edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Find the contour with the longest length that can be represented by a spline
max_length = 0
horizon_contour = None
for cnt in contours:
    # Check if the contour has enough points to be represented by a spline
    if len(cnt) >= 50:
        # Fit a spline to the contour
        x, y = cnt.squeeze().T
        tck, u = splprep([x, y], s=0, k=min(3, len(x)-1))

        # Compute the length of the spline
        length = 0
        for i in range(len(x)-1):
            p1 = splev(u[i], tck)
            p2 = splev(u[i+1], tck)
            length += np.linalg.norm(np.array(p2)-np.array(p1))

        # Check if the spline is the longest so far and doesn't cross over the same x coordinate more than once
        if length > max_length:
            max_length = length
            spline_x = splev(np.linspace(0, 1, 1000), tck)[0]
            if len(spline_x) == len(set(np.round(spline_x, 2))):
                horizon_contour = cnt

# Draw the contour on the image
cv2.drawContours(img, [horizon_contour], -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Horizon contour', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
