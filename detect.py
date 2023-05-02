import cv2
import numpy as np
from scipy.interpolate import splprep, splev

# Define minimum and maximum average height of the horizon spline (in percent of the image height, 0.0-1.0 = 0%-100%)
MIN_HEIGHT = 0.3
MAX_HEIGHT = 0.9

# Define weights for scoring
LENGTH_WEIGHT = 1.0
SMOOTHNESS_WEIGHT = 1.0
LINEARITY_WEIGHT = 1.0

# Load the image
img = cv2.imread(r'examples\R0011200.JPG')

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

# Best contour
horizon_contour = None

# Store all potential contours as tuple of (contour, score)
potential_contours = []

# Store all splines for debugging purposes
splines = []

# Best score
max_score = 0

for cnt in contours:
    # Check if the contour has enough points to be represented by a spline
    if len(cnt) >= 50:
        score = 0

        # Get the average height of the contour
        avg_height = np.mean(cnt[:, 0, 1])
        # Skip the contour if it's not in the right height range
        if avg_height < MIN_HEIGHT * img.shape[0] or avg_height > MAX_HEIGHT * img.shape[0]:
            continue

        # Fit a spline to the contour
        x, y = cnt.squeeze().T
        tck, u = splprep([x, y], s=len(cnt), k=1)

        # Store the spline for debugging purposes
        splines.append((tck, u))

        # Compute the length of the contour across the x-axis
        x_dist = max(x) - min(x)
        # Assign a score based on percentage of the image width
        score += LENGTH_WEIGHT * (x_dist / img.shape[1])

        # Compute the smoothness and linearity of the spline
        smoothness = 0
        linearity = 0
        for i in range(len(x)-1):
            p1 = splev(u[i], tck)
            p2 = splev(u[i+1], tck)
            smoothness += np.linalg.norm(np.array(p2)-np.array(p1))
            linearity += abs(p2[1] - p1[1]) / smoothness
        # Assign a score based on the linearity and smoothness of the spline
        score += SMOOTHNESS_WEIGHT * (smoothness / len(x))
        score += LINEARITY_WEIGHT * (linearity / len(x))

        # Add the contour to the list of potential contours
        potential_contours.append((cnt, score))

        # Check if the score is better than the previous best
        if score > max_score:
            max_score = score
            horizon_contour = cnt

# Find and print highest and lowest points of the horizon contour
min_y = min(horizon_contour[:, 0, 1])
max_y = max(horizon_contour[:, 0, 1])
print('Lowest point: ({}, {})'.format(horizon_contour[np.argmin(horizon_contour[:, 0, 1]), 0, 0], min_y))
print('Highest point: ({}, {})'.format(horizon_contour[np.argmax(horizon_contour[:, 0, 1]), 0, 0], max_y))

# Remove found contour from potential contours
potential_contours.remove((horizon_contour, max_score))

# Draw potential contours on the image
cv2.drawContours(img, [cnt for cnt, score in potential_contours], -1, (0, 255, 255), 2)

# Draw splines on the image
for tck, u in splines:
    x, y = splev(np.linspace(0, 1, 100), tck)
    cv2.polylines(img, np.int32([np.column_stack((x, y))]), False, (0, 0, 255), 2)

# Draw score of potential contours over each contour
for cnt, score in potential_contours:
    cv2.putText(img, str(round(score, 2)), tuple(cnt.squeeze()[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# Draw the contour on the image
cv2.drawContours(img, [horizon_contour], -1, (0, 255, 0), 2)

# Draw the score over the contour
cv2.putText(img, str(round(max_score, 2)), tuple(horizon_contour.squeeze()[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Display the result
cv2.imshow('Horizon contour', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
