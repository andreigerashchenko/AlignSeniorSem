import cv2
import numpy as np
from scipy.interpolate import splprep, splev

def score_candidate(img, tck, u, x, y, cnt, LENGTH_WEIGHT, SMOOTHNESS_WEIGHT, LINEARITY_WEIGHT):
    """Compute a score for a potential horizon contour."""
    score = 0

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

    return score

def find_horizon_point(img, LENGTH_WEIGHT, SMOOTHNESS_WEIGHT, LINEARITY_WEIGHT, MIN_HEIGHT, MAX_HEIGHT):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Blur the image to reduce noise
    gray = cv2.GaussianBlur(gray, (9, 15), 0)

    # Apply thresholding
    ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # Apply canny edge detection
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)

    # Find contours
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Best contour
    horizon_contour = None
    # Best spline
    horizon_spline = None

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

            # Compute a score for the contour
            score = score_candidate(img, tck, u, x, y, cnt, LENGTH_WEIGHT, SMOOTHNESS_WEIGHT, LINEARITY_WEIGHT)

            # Add the contour to the list of potential contours
            potential_contours.append((cnt, score))

            # Check if the score is better than the previous best
            if score > max_score:
                max_score = score
                horizon_contour = cnt
                horizon_spline = (tck, u)

    # Check if we found a horizon contour
    if horizon_contour is None:
        return None, None, None

    # Find highest and lowest points of the horizon spline
    min_x = horizon_contour[np.argmin(horizon_contour[:, 0, 1]), 0, 0]
    min_y = horizon_contour[np.argmin(horizon_contour[:, 0, 1]), 0, 1]
    max_x = horizon_contour[np.argmax(horizon_contour[:, 0, 1]), 0, 0]
    max_y = horizon_contour[np.argmax(horizon_contour[:, 0, 1]), 0, 1]

    # Find derivative at highest and lowest points of the horizon spline
    min_deriv = splev(horizon_spline[1][np.argmin(horizon_spline[1])], horizon_spline[0], der=1)[1]
    max_deriv = splev(horizon_spline[1][np.argmax(horizon_spline[1])], horizon_spline[0], der=1)[1]

    # Choose the point with the derivative closest to 0
    if abs(min_deriv) < abs(max_deriv):
        return min_x, min_y, "highest"
    else:
        return max_x, max_y, "lowest"