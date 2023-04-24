import cv2
import numpy as np
from scipy.interpolate import splprep, splev

# Define minimum and maximum average height of the horizon spline (in percent of the image height)
MIN_HEIGHT = 0.3 # 30%
MAX_HEIGHT = 0.9 # 90%

# Define roughness threshold
ROUGHNESS_THRESHOLD = 0.1

# Load the image
img = cv2.imread('examples\R0011132.JPG')

# Resize the image if it's larger than 1280x720
if img.shape[0] > 720 or img.shape[1] > 1280:
    scale_factor = min(1280 / img.shape[1], 720 / img.shape[0])
    img = cv2.resize(img, None, fx=scale_factor, fy=scale_factor)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur the image to reduce noise
gray = cv2.GaussianBlur(gray, (9, 15), 0)

# Apply thresholding
ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Display image before edge detection
cv2.imshow('Edge detection input', thresh)
cv2.waitKey(0)

# Apply canny edge detection
edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

# Find contours
contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Find the contour with the longest length that can be represented by a spline
max_length = 0
horizon_contour = None
x_crossings = 0

# Store all potential contours for debugging
potential_contours = []

for cnt in contours:
    # Check if the contour has enough points to be represented by a spline
    if len(cnt) >= 50:
        # Get the average height of the contour
        avg_height = np.mean(cnt[:, 0, 1])
        # Skip the contour if it's not in the right height range
        if avg_height < MIN_HEIGHT * img.shape[0] or avg_height > MAX_HEIGHT * img.shape[0]:
            continue

        # Check if the contour is too rough


        # Found a contour that might work
        potential_contours.append(cnt)

        # Fit a spline to the contour
        x, y = cnt.squeeze().T
        tck, u = splprep([x, y], s=0, k=min(3, len(x)-1))

        # Compute the length of the spline
        length = 0
        for i in range(len(x)-1):
            p1 = splev(u[i], tck)
            p2 = splev(u[i+1], tck)
            length += np.linalg.norm(np.array(p2)-np.array(p1))

        # Check if the spline is the longest so far and doesn't cross over the same X coordinate more often than the previous best
        if length > max_length:
            # Count the number of times the spline crosses the same X coordinate
            crossings = 0
            for i in range(len(x)-1):
                p1 = splev(u[i], tck)
                p2 = splev(u[i+1], tck)
                if p1[0] < 0.5 * img.shape[1] and p2[0] > 0.5 * img.shape[1] or p1[0] > 0.5 * img.shape[1] and p2[0] < 0.5 * img.shape[1]:
                    crossings += 1

            # Only save the spline if it crosses the same X coordinate less often than the previous best
            if crossings <= x_crossings:
                x_crossings = crossings
                max_length = length
                horizon_contour = cnt

# Remove found contour from potential contours
potential_contours.remove(horizon_contour)

# Draw potential contours on the image
cv2.drawContours(img, potential_contours, -1, (0, 0, 255), 2)

# Draw the contour on the image
cv2.drawContours(img, [horizon_contour], -1, (0, 255, 0), 2)

# Display the result
cv2.imshow('Horizon contour', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
